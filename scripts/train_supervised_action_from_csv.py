#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 模型
# =========================

class ActionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 57,
        output_dim: int = 20,
        hidden_dims=(512, 256, 128),
    ):
        super().__init__()

        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ELU())
            last_dim = h

        layers.append(nn.Linear(last_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NormalizedPolicy(nn.Module):
    """
    部署用模型：
        输入原始 obs
        内部自动归一化
        输出原始 action
    """

    def __init__(
        self,
        model: nn.Module,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ):
        super().__init__()
        self.model = model

        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        y_norm = self.model(x_norm)
        y = y_norm * self.target_std + self.target_mean
        return y


# =========================
# Dataset
# =========================

class CSVDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# =========================
# 数据处理
# =========================

def make_cols(prefix: str, dim: int):
    return [f"{prefix}_{i:02d}" for i in range(dim)]


def read_csv_auto(csv_path: str):
    """
    自动兼容逗号 CSV 或 tab 分隔文件。
    """
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    return df


def build_dataset_from_csv(
    csv_path: str,
    target_mode: str = "actions",
    use_group_split: bool = True,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    df = read_csv_auto(csv_path)

    print("[INFO] csv:", csv_path)
    print("[INFO] rows:", len(df))
    print("[INFO] cols:", len(df.columns))

    # -------- 输入列 --------
    ball_cols = make_cols("ball_center", 4)
    tactile_cols = make_cols("tactile_data", 13)
    joint_cols = make_cols("tesollo_joints_state", 20)
    action_cols = make_cols("actions", 20)

    required_cols = ball_cols + tactile_cols + joint_cols + action_cols

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少这些列: {missing}")

    # -------- 构造上一帧动作 previous_action --------
    # 注意：必须按 file_index 分组，不能让 data_0.pkl 的最后一帧影响 data_1.pkl 的第一帧
    if "file_index" in df.columns:
        prev_actions = df.groupby("file_index")[action_cols].shift(1)
    else:
        prev_actions = df[action_cols].shift(1)

    # 第一帧没有上一帧动作，填 0
    prev_actions = prev_actions.fillna(0.0)

    prev_action_cols = make_cols("previous_action", 20)
    prev_actions.columns = prev_action_cols

    # -------- 构造输入 x --------
    input_df = pd.concat(
        [
            df[ball_cols],
            df[tactile_cols],
            df[joint_cols],
            prev_actions,
        ],
        axis=1,
    )

    input_cols = ball_cols + tactile_cols + joint_cols + prev_action_cols

    # -------- 构造标签 y --------
    if target_mode == "actions":
        # 直接学习当前帧动作增量
        target_df = df[action_cols]

    elif target_mode == "next_actions":
        # 当前帧观测 -> 下一帧动作
        if "file_index" in df.columns:
            target_df = df.groupby("file_index")[action_cols].shift(-1)
        else:
            target_df = df[action_cols].shift(-1)

    elif target_mode == "next_joint_delta":
        # 当前帧观测 -> 下一帧关节角 - 当前帧关节角
        if "file_index" in df.columns:
            next_joint = df.groupby("file_index")[joint_cols].shift(-1)
        else:
            next_joint = df[joint_cols].shift(-1)

        target_df = next_joint.values - df[joint_cols].values
        target_df = pd.DataFrame(target_df, columns=action_cols)

    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    # -------- 转 numpy --------
    x_all = input_df.to_numpy(dtype=np.float32)
    y_all = target_df.to_numpy(dtype=np.float32)

    # 过滤 NaN/Inf
    finite_mask = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all).all(axis=1)

    x_all = x_all[finite_mask]
    y_all = y_all[finite_mask]

    if "file_index" in df.columns:
        file_index_all = df.loc[finite_mask, "file_index"].to_numpy()
    else:
        file_index_all = np.zeros(len(x_all), dtype=np.int32)

    print("[INFO] valid samples:", len(x_all))
    print("[INFO] input dim:", x_all.shape[1])
    print("[INFO] output dim:", y_all.shape[1])

    if x_all.shape[1] != 57:
        raise RuntimeError(f"输入维度应该是 57，但现在是 {x_all.shape[1]}")

    if y_all.shape[1] != 20:
        raise RuntimeError(f"输出维度应该是 20，但现在是 {y_all.shape[1]}")

    # -------- 划分训练集 / 验证集 --------
    rng = np.random.default_rng(seed)

    if use_group_split and "file_index" in df.columns:
        unique_files = np.unique(file_index_all)
        rng.shuffle(unique_files)

        num_val_files = max(1, int(len(unique_files) * val_ratio))
        val_files = set(unique_files[:num_val_files].tolist())

        val_mask = np.array([idx in val_files for idx in file_index_all])
        train_mask = ~val_mask

        print("[INFO] group split by file_index")
        print("[INFO] train files:", np.unique(file_index_all[train_mask]).shape[0])
        print("[INFO] val files:", np.unique(file_index_all[val_mask]).shape[0])

    else:
        indices = np.arange(len(x_all))
        rng.shuffle(indices)

        num_val = max(1, int(len(indices) * val_ratio))
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_mask = np.zeros(len(x_all), dtype=bool)
        val_mask = np.zeros(len(x_all), dtype=bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True

        print("[INFO] random row split")

    x_train = x_all[train_mask]
    y_train = y_all[train_mask]
    x_val = x_all[val_mask]
    y_val = y_all[val_mask]

    print("[INFO] train samples:", len(x_train))
    print("[INFO] val samples:", len(x_val))

    info = {
        "csv_path": csv_path,
        "target_mode": target_mode,
        "input_cols": input_cols,
        "target_cols": action_cols,
        "input_dim": 57,
        "output_dim": 20,
        "num_train": int(len(x_train)),
        "num_val": int(len(x_val)),
        "use_group_split": bool(use_group_split),
    }

    return x_train, y_train, x_val, y_val, info


# =========================
# 训练
# =========================

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, info = build_dataset_from_csv(
        csv_path=args.csv,
        target_mode=args.target_mode,
        use_group_split=not args.random_row_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # -------- 归一化 --------
    input_mean = torch.tensor(x_train.mean(axis=0), dtype=torch.float32)
    input_std = torch.tensor(x_train.std(axis=0), dtype=torch.float32).clamp_min(args.norm_eps)

    target_mean = torch.tensor(y_train.mean(axis=0), dtype=torch.float32)
    target_std = torch.tensor(y_train.std(axis=0), dtype=torch.float32).clamp_min(args.norm_eps)

    x_train_norm = (torch.tensor(x_train, dtype=torch.float32) - input_mean) / input_std
    y_train_norm = (torch.tensor(y_train, dtype=torch.float32) - target_mean) / target_std

    x_val_norm = (torch.tensor(x_val, dtype=torch.float32) - input_mean) / input_std
    y_val_norm = (torch.tensor(y_val, dtype=torch.float32) - target_mean) / target_std

    train_dataset = CSVDataset(x_train_norm.numpy(), y_train_norm.numpy())
    val_dataset = CSVDataset(x_val_norm.numpy(), y_val_norm.numpy())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # -------- 模型 --------
    hidden_dims = tuple(args.hidden_dims)
    model = ActionMLP(
        input_dim=57,
        output_dim=20,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Huber loss 比 MSE 对实机异常点更稳
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best_val_loss = float("inf")
    best_state = None

    # -------- 训练循环 --------
    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.shape[0]
            train_count += batch_x.shape[0]

        train_loss = train_loss_sum / max(1, train_count)

        # -------- 验证 --------
        model.eval()

        val_loss_sum = 0.0
        val_count = 0
        val_mae_raw_sum = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred_norm = model(batch_x)
                loss = loss_fn(pred_norm, batch_y)

                pred_raw = pred_norm.cpu() * target_std + target_mean
                target_raw = batch_y.cpu() * target_std + target_mean

                mae_raw = torch.mean(torch.abs(pred_raw - target_raw))

                val_loss_sum += loss.item() * batch_x.shape[0]
                val_mae_raw_sum += mae_raw.item() * batch_x.shape[0]
                val_count += batch_x.shape[0]

        val_loss = val_loss_sum / max(1, val_count)
        val_mae_raw = val_mae_raw_sum / max(1, val_count)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"[TRAIN] epoch={epoch:04d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mae_raw={val_mae_raw:.6f}"
            )

    # -------- 保存最优模型 --------
    if best_state is not None:
        model.load_state_dict(best_state)

    model_cpu = model.cpu().eval()

    checkpoint = {
        "model_state_dict": model_cpu.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "dataset_info": info,
        "hidden_dims": hidden_dims,
        "args": vars(args),
    }

    ckpt_path = output_dir / "supervised_action_policy.ckpt"
    torch.save(checkpoint, ckpt_path)

    # -------- 保存 TorchScript 部署模型 --------
    deploy_policy = NormalizedPolicy(
        model=model_cpu,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    ).eval()

    jit_path = output_dir / "supervised_action_policy_jit.pt"
    scripted = torch.jit.script(deploy_policy)
    scripted.save(jit_path)

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "dataset_info": info,
        "checkpoint": str(ckpt_path),
        "jit": str(jit_path),
        "input_order": info["input_cols"],
        "target_order": info["target_cols"],
        "note": "Input is raw 57-dim obs. TorchScript model internally normalizes input and denormalizes output.",
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("[DONE] saved checkpoint:", ckpt_path)
    print("[DONE] saved jit model:", jit_path)
    print("[DONE] saved metadata:", metadata_path)
    print("[DONE] best val loss:", best_val_loss)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/data/replay_data_0615_30HZ_1/replay_data_0615_30HZ_1.csv",
        help="整理后的 CSV 路径",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/logs/supervised_action_policy",
        help="模型输出目录",
    )

    parser.add_argument(
        "--target_mode",
        type=str,
        default="actions",
        choices=["actions", "next_actions", "next_joint_delta"],
        help=(
            "actions: 当前观测 -> 当前动作增量；"
            "next_actions: 当前观测 -> 下一帧动作；"
            "next_joint_delta: 当前观测 -> 下一帧关节角 - 当前关节角"
        ),
    )

    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--huber_beta", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--norm_eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument(
        "--random_row_split",
        action="store_true",
        help="默认按 file_index 分组划分训练/验证；加这个参数则随机按行划分。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())