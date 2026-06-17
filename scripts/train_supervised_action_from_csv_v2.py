#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


BALL_DIM = 4
TACTILE_DIM = 13
JOINT_DIM = 20
ACTION_DIM = 20
INPUT_DIM = BALL_DIM + TACTILE_DIM + JOINT_DIM + ACTION_DIM


def make_cols(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(dim)]


def read_csv_auto(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    return df


def detect_group_col(df: pd.DataFrame) -> str | None:
    for name in ("file_index", "episode", "episode_id", "traj_id", "trajectory_id"):
        if name in df.columns:
            return name
    return None


@dataclass
class InputPreprocessCfg:
    ball_normalization: str
    camera_width: int
    camera_height: int


class ActionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeployPolicy(nn.Module):
    """TorchScript wrapper.

    The scripted policy expects the raw 57-dim observation in metadata input
    order. It applies the same ball preprocessing and z-score normalization used
    during training, and returns action values in the original target units.
    """

    def __init__(
        self,
        model: nn.Module,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        ball_normalization_mode: int,
        camera_width: int,
        camera_height: int,
    ):
        super().__init__()
        self.model = model
        self.ball_normalization_mode = int(ball_normalization_mode)
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("camera_width_minus_one", torch.tensor(float(max(camera_width - 1, 1))))
        self.register_buffer("camera_height_minus_one", torch.tensor(float(max(camera_height - 1, 1))))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.ball_normalization_mode != 1:
            return x

        x_proc = x.clone()
        x_proc[..., 0] = 2.0 * x_proc[..., 0] / self.camera_width_minus_one - 1.0
        x_proc[..., 1] = 2.0 * x_proc[..., 1] / self.camera_height_minus_one - 1.0
        x_proc[..., 2] = 2.0 * x_proc[..., 2] / self.camera_width_minus_one - 1.0
        x_proc[..., 3] = 2.0 * x_proc[..., 3] / self.camera_height_minus_one - 1.0
        return x_proc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proc = self._preprocess(x)
        x_norm = (x_proc - self.input_mean) / self.input_std
        y_norm = self.model(x_norm)
        return y_norm * self.target_std + self.target_mean


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def build_previous_actions(df: pd.DataFrame, group_col: str | None, action_cols: list[str]) -> pd.DataFrame:
    if group_col is not None:
        prev_actions = df.groupby(group_col, sort=False)[action_cols].shift(1)
    else:
        prev_actions = df[action_cols].shift(1)

    prev_actions = prev_actions.fillna(0.0)
    prev_actions.columns = make_cols("previous_action", ACTION_DIM)
    return prev_actions


def build_targets(
    df: pd.DataFrame,
    group_col: str | None,
    target_mode: str,
    action_cols: list[str],
    joint_cols: list[str],
) -> pd.DataFrame:
    if target_mode == "actions":
        return df[action_cols]

    if target_mode == "next_actions":
        if group_col is not None:
            return df.groupby(group_col, sort=False)[action_cols].shift(-1)
        return df[action_cols].shift(-1)

    if target_mode == "next_joint_delta":
        if group_col is not None:
            next_joint = df.groupby(group_col, sort=False)[joint_cols].shift(-1)
        else:
            next_joint = df[joint_cols].shift(-1)
        target_np = next_joint.to_numpy(dtype=np.float32) - df[joint_cols].to_numpy(dtype=np.float32)
        return pd.DataFrame(target_np, columns=action_cols, index=df.index)

    raise ValueError(f"Unknown target_mode: {target_mode}")


def preprocess_inputs(x: np.ndarray, cfg: InputPreprocessCfg) -> np.ndarray:
    x_proc = x.astype(np.float32, copy=True)
    if cfg.ball_normalization == "pixel_to_uv":
        width_minus_one = max(float(cfg.camera_width - 1), 1.0)
        height_minus_one = max(float(cfg.camera_height - 1), 1.0)
        x_proc[:, 0] = 2.0 * x_proc[:, 0] / width_minus_one - 1.0
        x_proc[:, 1] = 2.0 * x_proc[:, 1] / height_minus_one - 1.0
        x_proc[:, 2] = 2.0 * x_proc[:, 2] / width_minus_one - 1.0
        x_proc[:, 3] = 2.0 * x_proc[:, 3] / height_minus_one - 1.0
    elif cfg.ball_normalization != "none":
        raise ValueError(f"Unknown ball_normalization: {cfg.ball_normalization}")
    return x_proc


def build_split_masks(
    row_indices: np.ndarray,
    groups: np.ndarray | None,
    split_mode: str,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(row_indices) < 2:
        raise RuntimeError("Need at least two valid samples to create train/val split.")

    rng = np.random.default_rng(seed)
    resolved_mode = split_mode
    if resolved_mode == "auto":
        resolved_mode = "group" if groups is not None and len(np.unique(groups)) > 1 else "time"

    if resolved_mode == "group" and (groups is None or len(np.unique(groups)) <= 1):
        print("[WARN] group split requested but no multiple groups were found; falling back to time split.")
        resolved_mode = "time"

    train_mask = np.zeros(len(row_indices), dtype=bool)
    val_mask = np.zeros(len(row_indices), dtype=bool)
    split_info: dict = {"split_mode": resolved_mode, "val_ratio": float(val_ratio)}

    if resolved_mode == "group":
        assert groups is not None
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        num_val_groups = max(1, int(round(len(unique_groups) * val_ratio)))
        num_val_groups = min(num_val_groups, len(unique_groups) - 1)
        val_groups = set(unique_groups[:num_val_groups].tolist())
        val_mask = np.array([group in val_groups for group in groups], dtype=bool)
        train_mask = ~val_mask
        split_info["val_groups"] = sorted([str(group) for group in val_groups])

    elif resolved_mode == "time":
        if groups is None:
            num_val = max(1, int(round(len(row_indices) * val_ratio)))
            num_val = min(num_val, len(row_indices) - 1)
            val_mask[-num_val:] = True
            train_mask[:-num_val] = True
        else:
            for group in np.unique(groups):
                group_ids = np.where(groups == group)[0]
                if len(group_ids) <= 1:
                    train_mask[group_ids] = True
                    continue
                num_val = max(1, int(round(len(group_ids) * val_ratio)))
                num_val = min(num_val, len(group_ids) - 1)
                val_mask[group_ids[-num_val:]] = True
                train_mask[group_ids[:-num_val]] = True

    elif resolved_mode == "random":
        indices = np.arange(len(row_indices))
        rng.shuffle(indices)
        num_val = max(1, int(round(len(indices) * val_ratio)))
        num_val = min(num_val, len(indices) - 1)
        val_mask[indices[:num_val]] = True
        train_mask[indices[num_val:]] = True
        split_info["warning"] = "random row split can leak temporal information for time-series data"

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if not train_mask.any() or not val_mask.any():
        raise RuntimeError(f"Invalid split: train={train_mask.sum()} val={val_mask.sum()}")

    split_info["num_train"] = int(train_mask.sum())
    split_info["num_val"] = int(val_mask.sum())
    return train_mask, val_mask, split_info


def build_dataset_from_csv(args: argparse.Namespace):
    df = read_csv_auto(args.csv)
    group_col = detect_group_col(df)

    ball_cols = make_cols("ball_center", BALL_DIM)
    tactile_cols = make_cols("tactile_data", TACTILE_DIM)
    joint_cols = make_cols("tesollo_joints_state", JOINT_DIM)
    action_cols = make_cols("actions", ACTION_DIM)

    required_cols = ball_cols + tactile_cols + joint_cols + action_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"CSV is missing required columns: {missing}")

    prev_actions = build_previous_actions(df, group_col, action_cols)
    input_df = pd.concat([df[ball_cols], df[tactile_cols], df[joint_cols], prev_actions], axis=1)
    target_df = build_targets(df, group_col, args.target_mode, action_cols, joint_cols)

    x_raw = input_df.to_numpy(dtype=np.float32)
    y_raw = target_df.to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(x_raw).all(axis=1) & np.isfinite(y_raw).all(axis=1)

    row_indices = np.where(finite_mask)[0]
    x_raw = x_raw[finite_mask]
    y_raw = y_raw[finite_mask]

    groups = None
    if group_col is not None:
        groups = df.loc[finite_mask, group_col].to_numpy()

    preprocess_cfg = InputPreprocessCfg(
        ball_normalization=args.ball_normalization,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    x_model = preprocess_inputs(x_raw, preprocess_cfg)

    train_mask, val_mask, split_info = build_split_masks(
        row_indices=row_indices,
        groups=groups,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    info = {
        "csv_path": str(args.csv),
        "rows": int(len(df)),
        "valid_samples": int(len(x_raw)),
        "group_col": group_col,
        "target_mode": args.target_mode,
        "target_note": {
            "actions": "current observation -> current recorded action",
            "next_actions": "current observation -> next recorded action",
            "next_joint_delta": "current observation -> next joint position minus current joint position",
        }[args.target_mode],
        "raw_input_cols": ball_cols + tactile_cols + joint_cols + make_cols("previous_action", ACTION_DIM),
        "target_cols": action_cols,
        "input_dim": INPUT_DIM,
        "output_dim": ACTION_DIM,
        "input_preprocess": asdict(preprocess_cfg),
        "split": split_info,
    }

    print("[INFO] csv:", args.csv)
    print("[INFO] rows:", len(df))
    print("[INFO] valid samples:", len(x_raw))
    print("[INFO] group col:", group_col)
    print("[INFO] target mode:", args.target_mode)
    print("[INFO] split mode:", split_info["split_mode"])
    print("[INFO] train samples:", split_info["num_train"])
    print("[INFO] val samples:", split_info["num_val"])

    return {
        "x_train": x_model[train_mask],
        "y_train": y_raw[train_mask],
        "x_val": x_model[val_mask],
        "y_val": y_raw[val_mask],
        "x_val_raw": x_raw[val_mask],
        "row_indices_val": row_indices[val_mask],
        "info": info,
        "preprocess_cfg": preprocess_cfg,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    loss_sum = 0.0
    count = 0
    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_norm = model(batch_x)
            loss = loss_fn(pred_norm, batch_y)

            pred_raw = pred_norm.cpu() * target_std + target_mean
            target_raw = batch_y.cpu() * target_std + target_mean

            loss_sum += loss.item() * batch_x.shape[0]
            count += batch_x.shape[0]
            all_pred.append(pred_raw)
            all_target.append(target_raw)

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)
    err = pred - target
    abs_err = torch.abs(err)
    mae_per_joint = abs_err.mean(dim=0)
    rmse_per_joint = torch.sqrt(torch.mean(err * err, dim=0))

    return {
        "loss": loss_sum / max(1, count),
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(err * err)).item()),
        "max_abs": float(abs_err.max().item()),
        "mae_per_joint": mae_per_joint.numpy(),
        "rmse_per_joint": rmse_per_joint.numpy(),
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = requested_device
    print("[INFO] device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataset_from_csv(args)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    info = data["info"]
    preprocess_cfg = data["preprocess_cfg"]

    input_mean = torch.as_tensor(x_train.mean(axis=0), dtype=torch.float32)
    input_std = torch.as_tensor(x_train.std(axis=0), dtype=torch.float32).clamp_min(args.norm_eps)
    target_mean = torch.as_tensor(y_train.mean(axis=0), dtype=torch.float32)
    target_std = torch.as_tensor(y_train.std(axis=0), dtype=torch.float32).clamp_min(args.norm_eps)

    x_train_norm = (torch.as_tensor(x_train, dtype=torch.float32) - input_mean) / input_std
    y_train_norm = (torch.as_tensor(y_train, dtype=torch.float32) - target_mean) / target_std
    x_val_norm = (torch.as_tensor(x_val, dtype=torch.float32) - input_mean) / input_std
    y_val_norm = (torch.as_tensor(y_val, dtype=torch.float32) - target_mean) / target_std

    train_loader = DataLoader(
        ArrayDataset(x_train_norm.numpy(), y_train_norm.numpy()),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ArrayDataset(x_val_norm.numpy(), y_val_norm.numpy()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    hidden_dims = tuple(args.hidden_dims)
    model = ActionMLP(
        input_dim=INPUT_DIM,
        output_dim=ACTION_DIM,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best_val_loss = float("inf")
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0
    history = []

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
            if args.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.shape[0]
            train_count += batch_x.shape[0]

        train_loss = train_loss_sum / max(1, train_count)
        metrics = evaluate(model, val_loader, loss_fn, target_mean, target_std, device)
        scheduler.step(metrics["loss"])

        improved = metrics["loss"] < best_val_loss - args.min_delta
        if improved:
            best_val_loss = metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(metrics["loss"]),
                "val_mae": float(metrics["mae"]),
                "val_rmse": float(metrics["rmse"]),
                "val_max_abs": float(metrics["max_abs"]),
                "lr": current_lr,
            }
        )

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs or improved:
            worst_joint = int(np.argmax(metrics["mae_per_joint"]))
            print(
                f"[TRAIN] epoch={epoch:04d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={metrics['loss']:.6f} "
                f"val_mae={metrics['mae']:.6f} "
                f"worst_joint={worst_joint:02d}:{metrics['mae_per_joint'][worst_joint]:.6f} "
                f"lr={current_lr:.3e}"
            )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"[INFO] early stop at epoch {epoch}; best val loss={best_val_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_cpu = model.cpu().eval()

    ball_norm_mode = 1 if preprocess_cfg.ball_normalization == "pixel_to_uv" else 0
    deploy_policy = DeployPolicy(
        model=model_cpu,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        ball_normalization_mode=ball_norm_mode,
        camera_width=preprocess_cfg.camera_width,
        camera_height=preprocess_cfg.camera_height,
    ).eval()

    ckpt_path = output_dir / "supervised_action_policy_v2.ckpt"
    jit_path = output_dir / "supervised_action_policy_v2_jit.pt"
    metadata_path = output_dir / "metadata_v2.json"
    history_path = output_dir / "history_v2.json"
    per_joint_path = output_dir / "per_joint_metrics_v2.csv"

    checkpoint = {
        "model_state_dict": model_cpu.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "hidden_dims": hidden_dims,
        "args": vars(args),
        "dataset_info": info,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, ckpt_path)

    scripted = torch.jit.script(deploy_policy)
    scripted.save(jit_path)

    if best_metrics is None:
        best_metrics = evaluate(model_cpu.to(device), val_loader, loss_fn, target_mean, target_std, device)
        model_cpu.cpu()

    per_joint = pd.DataFrame(
        {
            "joint": make_cols("actions", ACTION_DIM),
            "mae": best_metrics["mae_per_joint"],
            "rmse": best_metrics["rmse_per_joint"],
        }
    )
    per_joint.to_csv(per_joint_path, index=False)

    metadata = {
        "dataset_info": info,
        "checkpoint": str(ckpt_path),
        "jit": str(jit_path),
        "history": str(history_path),
        "per_joint_metrics": str(per_joint_path),
        "model": {
            "input_dim": INPUT_DIM,
            "output_dim": ACTION_DIM,
            "hidden_dims": list(hidden_dims),
            "dropout": float(args.dropout),
            "use_layer_norm": bool(args.use_layer_norm),
        },
        "best_metrics": {
            "val_loss": float(best_val_loss),
            "val_mae": float(best_metrics["mae"]),
            "val_rmse": float(best_metrics["rmse"]),
            "val_max_abs": float(best_metrics["max_abs"]),
        },
        "deployment_contract": (
            "TorchScript expects raw 57-dim obs in raw_input_cols order. "
            "It applies input_preprocess and z-score normalization internally, "
            "then returns action in the original target units."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("[DONE] saved checkpoint:", ckpt_path)
    print("[DONE] saved jit model:", jit_path)
    print("[DONE] saved metadata:", metadata_path)
    print("[DONE] saved history:", history_path)
    print("[DONE] saved per-joint metrics:", per_joint_path)
    print("[DONE] best val loss:", best_val_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True, help="Path to replay CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and metadata.")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="next_actions",
        choices=["actions", "next_actions", "next_joint_delta"],
        help="Default next_actions avoids using post-action observations to predict the action that caused them.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="auto",
        choices=["auto", "group", "time", "random"],
        help="auto uses file_index groups when available, otherwise chronological time split.",
    )
    parser.add_argument(
        "--ball_normalization",
        type=str,
        default="pixel_to_uv",
        choices=["pixel_to_uv", "none"],
        help="pixel_to_uv maps ball_center pixels to [-1, 1] before z-score normalization.",
    )
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)

    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true")
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
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--scheduler_patience", type=int, default=12)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
