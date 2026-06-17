#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def make_cols(prefix: str, dim: int):
    return [f"{prefix}_{i:02d}" for i in range(dim)]


def read_csv_auto(csv_path: str):
    """
    自动兼容逗号 CSV / tab CSV。
    """
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    return df


def build_previous_actions(df: pd.DataFrame, action_cols: list[str]):
    """
    构造上一帧动作 previous_action。

    注意：
    必须按 file_index 分组，否则 data_0.pkl 最后一帧会串到 data_1.pkl 第一帧。
    """
    if "file_index" in df.columns:
        prev_actions = df.groupby("file_index")[action_cols].shift(1)
    else:
        prev_actions = df[action_cols].shift(1)

    prev_actions = prev_actions.fillna(0.0)
    return prev_actions


def build_targets(df: pd.DataFrame, target_mode: str):
    action_cols = make_cols("actions", 20)
    joint_cols = make_cols("tesollo_joints_state", 20)

    if target_mode == "actions":
        target_df = df[action_cols]

    elif target_mode == "next_actions":
        if "file_index" in df.columns:
            target_df = df.groupby("file_index")[action_cols].shift(-1)
        else:
            target_df = df[action_cols].shift(-1)

    elif target_mode == "next_joint_delta":
        if "file_index" in df.columns:
            next_joint = df.groupby("file_index")[joint_cols].shift(-1)
        else:
            next_joint = df[joint_cols].shift(-1)

        target_np = next_joint.to_numpy(dtype=np.float32) - df[joint_cols].to_numpy(dtype=np.float32)
        target_df = pd.DataFrame(target_np, columns=action_cols)

    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    return target_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/data/replay_data_0615_30HZ_1/replay_data_0615_30HZ_1.csv",
    )

    parser.add_argument(
        "--policy",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/logs/supervised_action_policy/supervised_action_policy_jit.pt",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/logs/supervised_action_policy/metadata.json",
    )

    parser.add_argument(
        "--target_mode",
        type=str,
        default="auto",
        choices=["auto", "actions", "next_actions", "next_joint_delta"],
    )

    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--save_result",
        type=str,
        default="",
        help="如果指定路径，会把预测结果保存成 CSV。",
    )

    args = parser.parse_args()

    # =========================
    # 读取 metadata，自动获取 target_mode
    # =========================
    target_mode = args.target_mode

    metadata_path = Path(args.metadata)
    if target_mode == "auto":
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            target_mode = metadata.get("dataset_info", {}).get("target_mode", "actions")
            print("[INFO] target_mode from metadata:", target_mode)
        else:
            target_mode = "actions"
            print("[WARN] metadata not found, use target_mode=actions")

    # =========================
    # 加载 CSV
    # =========================
    df = read_csv_auto(args.csv)

    print("[INFO] csv:", args.csv)
    print("[INFO] rows:", len(df))
    print("[INFO] cols:", len(df.columns))

    ball_cols = make_cols("ball_center", 4)
    tactile_cols = make_cols("tactile_data", 13)
    joint_cols = make_cols("tesollo_joints_state", 20)
    action_cols = make_cols("actions", 20)

    required_cols = ball_cols + tactile_cols + joint_cols + action_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少列: {missing}")

    # =========================
    # 构造输入 obs
    # =========================
    prev_actions = build_previous_actions(df, action_cols)
    prev_action_cols = make_cols("previous_action", 20)
    prev_actions.columns = prev_action_cols

    input_df = pd.concat(
        [
            df[ball_cols],
            df[tactile_cols],
            df[joint_cols],
            prev_actions,
        ],
        axis=1,
    )

    target_df = build_targets(df, target_mode)

    x_all = input_df.to_numpy(dtype=np.float32)
    y_all = target_df.to_numpy(dtype=np.float32)

    finite_mask = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all).all(axis=1)

    valid_indices = np.where(finite_mask)[0]
    if len(valid_indices) == 0:
        raise RuntimeError("没有有效样本，可能 actions 第一帧为空或 CSV 中有 NaN。")

    print("[INFO] valid samples:", len(valid_indices))
    print("[INFO] input dim:", x_all.shape[1])
    print("[INFO] target dim:", y_all.shape[1])

    # =========================
    # 随机抽样
    # =========================
    rng = np.random.default_rng(args.seed)

    num_samples = min(args.num_samples, len(valid_indices))
    chosen_indices = rng.choice(valid_indices, size=num_samples, replace=False)

    # 按原始顺序排序，方便看
    chosen_indices = np.sort(chosen_indices)

    x = x_all[chosen_indices]
    y_true = y_all[chosen_indices]

    # =========================
    # 加载模型
    # =========================
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    policy = torch.jit.load(args.policy, map_location=device)
    policy.eval()

    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_pred = policy(x_t).detach().cpu().numpy()

    # =========================
    # 计算误差
    # =========================
    err = y_pred - y_true
    abs_err = np.abs(err)

    mae_per_sample = abs_err.mean(axis=1)
    max_err_per_sample = abs_err.max(axis=1)

    overall_mae = abs_err.mean()
    overall_max = abs_err.max()
    overall_rmse = np.sqrt(np.mean(err ** 2))

    print("")
    print("========== Overall Error ==========")
    print(f"target_mode  : {target_mode}")
    print(f"num_samples  : {num_samples}")
    print(f"MAE          : {overall_mae:.8f}")
    print(f"RMSE         : {overall_rmse:.8f}")
    print(f"MAX_ABS_ERR  : {overall_max:.8f}")
    print("===================================")

    # =========================
    # 打印每个样本
    # =========================
    result_rows = []

    for n, row_idx in enumerate(chosen_indices):
        file_index = df.loc[row_idx, "file_index"] if "file_index" in df.columns else -1
        step = df.loc[row_idx, "step"] if "step" in df.columns else row_idx

        print("")
        print(f"---------- sample {n} ----------")
        print(f"csv_row    : {row_idx}")
        print(f"file_index : {file_index}")
        print(f"step       : {step}")
        print(f"mae        : {mae_per_sample[n]:.8f}")
        print(f"max_abs_err: {max_err_per_sample[n]:.8f}")

        print("true_action:")
        print(np.array2string(y_true[n], precision=5, suppress_small=False))

        print("pred_action:")
        print(np.array2string(y_pred[n], precision=5, suppress_small=False))

        print("error:")
        print(np.array2string(err[n], precision=5, suppress_small=False))

        row = {
            "csv_row": int(row_idx),
            "file_index": int(file_index) if file_index != -1 else -1,
            "step": int(step),
            "mae": float(mae_per_sample[n]),
            "max_abs_err": float(max_err_per_sample[n]),
        }

        for i in range(20):
            row[f"true_action_{i:02d}"] = float(y_true[n, i])
            row[f"pred_action_{i:02d}"] = float(y_pred[n, i])
            row[f"err_action_{i:02d}"] = float(err[n, i])

        result_rows.append(row)

    # =========================
    # 保存结果
    # =========================
    if args.save_result:
        save_path = Path(args.save_result)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(save_path, index=False)

        print("")
        print("[DONE] saved result:", save_path)


if __name__ == "__main__":
    main()
