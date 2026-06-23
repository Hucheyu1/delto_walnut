#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# Utils
# ============================================================

def make_cols(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(dim)]


def read_csv_auto(csv_path: str | Path) -> pd.DataFrame:
    """
    自动兼容逗号 CSV / tab CSV。
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    return df


def add_history_suffix(policy_name: str, use_history_stack: bool) -> str:
    if use_history_stack and not policy_name.endswith("_history"):
        return policy_name + "_history"
    return policy_name


def default_policy_name_from_obs_mode(obs_mode: str) -> str:
    if obs_mode == "full":
        return "supervised_action_policy_full"
    if obs_mode == "no_vision":
        return "supervised_action_policy_no_vision"
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def add_target_mode_suffix(filename: str, target_mode: str):
    path = Path(filename)
    return f"{path.stem}_{target_mode}{path.suffix}"


# ============================================================
# Previous action / Target
# ============================================================

def build_previous_actions(df: pd.DataFrame, action_cols: list[str]) -> pd.DataFrame:
    """
    构造 previous_action。

    测试单个 CSV 时按 file_index 分组即可，避免 data_0.pkl 串到 data_1.pkl。
    """
    missing = [c for c in action_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少 actions 列，无法构造 previous_action: {missing}")

    if "file_index" in df.columns:
        prev_actions = df.groupby("file_index")[action_cols].shift(1)
    else:
        prev_actions = df[action_cols].shift(1)

    prev_actions = prev_actions.fillna(0.0)
    return prev_actions


def build_targets(df: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    action_cols = make_cols("actions", 20)
    joint_cols = make_cols("tesollo_joints_state", 20)

    missing_action = [c for c in action_cols if c not in df.columns]
    if missing_action:
        raise KeyError(f"CSV 缺少 actions 列: {missing_action}")

    if target_mode == "actions":
        target_df = df[action_cols]

    elif target_mode == "action_delta":
        prev_actions = build_previous_actions(df, action_cols)
        target_np = df[action_cols].to_numpy(dtype=np.float32) - prev_actions.to_numpy(dtype=np.float32)
        target_df = pd.DataFrame(target_np, columns=action_cols)

    elif target_mode == "next_actions":
        if "file_index" in df.columns:
            target_df = df.groupby("file_index")[action_cols].shift(-1)
        else:
            target_df = df[action_cols].shift(-1)

    elif target_mode == "next_joint_delta":
        missing_joint = [c for c in joint_cols if c not in df.columns]
        if missing_joint:
            raise KeyError(f"CSV 缺少关节列，无法构造 next_joint_delta: {missing_joint}")

        if "file_index" in df.columns:
            next_joint = df.groupby("file_index")[joint_cols].shift(-1)
        else:
            next_joint = df[joint_cols].shift(-1)

        target_np = next_joint.to_numpy(dtype=np.float32) - df[joint_cols].to_numpy(dtype=np.float32)
        target_df = pd.DataFrame(target_np, columns=action_cols)

    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    return target_df


# ============================================================
# Metadata / Path resolving
# ============================================================

TARGET_MODES = ["actions", "action_delta", "next_actions", "next_joint_delta"]


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        print("[WARN] metadata not found:", metadata_path)
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def resolve_target_mode(args_target_mode: str, metadata: dict) -> str:
    if args_target_mode != "auto":
        return args_target_mode

    target_mode = (
        metadata.get("dataset_info", {}).get("target_mode", None)
        or metadata.get("target_mode", None)
    )

    if target_mode is None:
        target_mode = "actions"
        print("[WARN] target_mode not found in metadata, use actions")
    else:
        print("[INFO] target_mode from metadata:", target_mode)

    return target_mode


def find_metadata_for_auto(policy_dir: Path):
    candidates = []

    for mode in TARGET_MODES:
        p = policy_dir / mode / "metadata.json"
        if p.exists():
            metadata = load_metadata(p)
            actual_mode = resolve_target_mode("auto", metadata)
            candidates.append((actual_mode, p))

    # 兼容老结构：metadata 直接在 policy_dir 下
    p = policy_dir / "metadata.json"
    if p.exists():
        metadata = load_metadata(p)
        mode = resolve_target_mode("auto", metadata)
        candidates.append((mode, p))

    if len(candidates) == 0:
        raise FileNotFoundError(
            "target_mode=auto，但没有找到 metadata.json。\n"
            f"已搜索目录: {policy_dir}\n"
            "请显式指定 --target_mode actions/action_delta/next_actions/next_joint_delta，"
            "或者用 --metadata 指定 metadata.json 路径。"
        )

    unique = []
    seen_paths = set()
    for mode, path in candidates:
        if str(path) not in seen_paths:
            unique.append((mode, path))
            seen_paths.add(str(path))

    if len(unique) > 1:
        msg = "\n".join([f"  {mode}: {path}" for mode, path in unique])
        raise RuntimeError(
            "target_mode=auto，但发现多个 metadata.json，无法判断该测试哪一个：\n"
            f"{msg}\n\n"
            "请显式指定，例如：\n"
            "  --target_mode action_delta\n"
            "或者：\n"
            "  --target_mode actions"
        )

    return unique[0]


def resolve_policy_name(obs_mode: str, policy_name: str, use_history_stack: bool) -> str:
    if policy_name:
        base_name = policy_name
    else:
        base_name = default_policy_name_from_obs_mode(obs_mode)

    return add_history_suffix(base_name, use_history_stack)


def resolve_paths(args: argparse.Namespace):
    """
    目录结构默认采用训练脚本输出格式：
        logs/{policy_name}/{target_mode}/metadata.json
        logs/{policy_name}/{target_mode}/supervised_action_policy_jit.pt

    历史模型：
        加 --use_history_stack 后，policy_name 自动追加 _history。
    """
    args.policy_name = resolve_policy_name(
        obs_mode=args.obs_mode,
        policy_name=args.policy_name,
        use_history_stack=args.use_history_stack,
    )

    policy_dir = Path(args.log_root) / args.policy_name

    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata = load_metadata(metadata_path)
        target_mode = resolve_target_mode(args.target_mode, metadata)
    else:
        if args.target_mode == "auto":
            target_mode, metadata_path = find_metadata_for_auto(policy_dir)
            metadata = load_metadata(metadata_path)
            target_mode = resolve_target_mode("auto", metadata)
        else:
            target_mode = args.target_mode
            metadata_path = policy_dir / target_mode / "metadata.json"
            metadata = load_metadata(metadata_path)

    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = policy_dir / target_mode

    if args.policy:
        policy_path = Path(args.policy)
    else:
        policy_path = None

        # 1. metadata 里的 default_jit
        default_jit_from_metadata = metadata.get("default_jit", "")
        if default_jit_from_metadata:
            p = Path(default_jit_from_metadata)
            if p.exists():
                policy_path = p

        # 2. metadata 里的 jit
        if policy_path is None:
            jit_from_metadata = metadata.get("jit", "")
            if jit_from_metadata:
                p = Path(jit_from_metadata)
                if p.exists():
                    policy_path = p

        # 3. 默认名字
        if policy_path is None:
            default_jit = policy_dir / target_mode / "supervised_action_policy_jit.pt"
            if default_jit.exists():
                policy_path = default_jit

        # 4. 兼容 supervised_{target_mode}_policy_jit.pt
        if policy_path is None:
            mode_jit = policy_dir / target_mode / f"supervised_{target_mode}_policy_jit.pt"
            if mode_jit.exists():
                policy_path = mode_jit

        # 5. 即使不存在，也给出默认路径，后面统一报错
        if policy_path is None:
            policy_path = policy_dir / target_mode / "supervised_action_policy_jit.pt"

    args.target_mode = target_mode
    args.result_dir = str(result_dir)
    args.metadata = str(metadata_path)
    args.policy = str(policy_path)

    return policy_dir, result_dir, metadata_path, policy_path, metadata, target_mode


def resolve_input_cols(metadata: dict, args: argparse.Namespace) -> list[str]:
    """
    优先从 metadata 读取训练时的 input_order。

    如果没有 metadata/input_order，则按命令行参数兜底构造。
    """
    input_cols = metadata.get("input_order", None)
    if input_cols is None:
        input_cols = metadata.get("dataset_info", {}).get("input_cols", None)

    if input_cols is not None:
        print("[INFO] input_order from metadata")
        print("[INFO] input dim:", len(input_cols))
        return input_cols

    print("[WARN] input_order not found in metadata, fallback by args")

    ball_cols = make_cols("ball_center", 4)
    tactile_cols = make_cols("tactile_data", 13)
    joint_cols = make_cols("tesollo_joints_state", 20)
    prev_action_cols = make_cols("previous_action", 20)

    if args.obs_mode == "full":
        base_cols = ball_cols + tactile_cols + joint_cols
    elif args.obs_mode == "no_vision":
        base_cols = tactile_cols + joint_cols
    else:
        raise ValueError(f"Unknown obs_mode: {args.obs_mode}")

    if args.use_history_stack:
        cols = []
        for lag in range(args.history_len - 1, -1, -1):
            cols.extend([f"hist_{lag:02d}_{c}" for c in base_cols])
        if args.history_include_previous_action:
            cols.extend(prev_action_cols)
        return cols

    return base_cols + prev_action_cols


# ============================================================
# Input building, compatible with old and history models
# ============================================================

def ensure_global_traj_id_for_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "global_traj_id" not in df.columns:
        if "file_index" in df.columns:
            df["global_traj_id"] = "csv::file_" + df["file_index"].astype(str)
        else:
            df["file_index"] = 0
            df["global_traj_id"] = "csv::file_0"
    return df


def parse_history_cols(input_cols: list[str]) -> tuple[bool, int, list[str]]:
    """
    从 input_order 中判断是否为历史模型，并推断 history_len 和 base_cols。

    hist_09_tactile_data_00 -> lag=9, base_col=tactile_data_00
    """
    is_history = any(c.startswith("hist_") for c in input_cols)
    if not is_history:
        return False, 1, []

    max_lag = 0
    base_cols = []
    seen = set()

    for c in input_cols:
        if not c.startswith("hist_"):
            continue

        parts = c.split("_", 2)
        if len(parts) != 3:
            raise ValueError(f"非法历史输入列名: {c}")

        lag = int(parts[1])
        base_col = parts[2]
        max_lag = max(max_lag, lag)

        # 只用最老一帧的列顺序还原 base_cols，避免重复加入 10 次。
        if base_col not in seen:
            base_cols.append(base_col)
            seen.add(base_col)

    history_len = max_lag + 1
    return True, history_len, base_cols


def build_history_stacked_features(
    df: pd.DataFrame,
    base_cols: list[str],
    group_col: str,
    history_len: int,
) -> pd.DataFrame:
    """
    历史堆叠输入。

    history_len=10 时：
        [t-9, t-8, ..., t-1, t]

    轨迹开头不足 history_len 的部分不补齐，直接保留 NaN。
    后续会通过 finite_mask 自动过滤掉这些样本。

    例如 history_len=10：
        每条轨迹的前 9 帧都会被舍弃。
    """
    if history_len <= 0:
        raise ValueError(f"history_len must be > 0, got {history_len}")

    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少历史堆叠需要的列: {missing}")

    if group_col not in df.columns:
        raise KeyError(f"CSV 缺少分组列: {group_col}")

    parts = []

    # 从旧到新排列：
    # history_len=10 时：
    # lag=9 -> t-9
    # lag=8 -> t-8
    # ...
    # lag=0 -> t
    for lag in range(history_len - 1, -1, -1):
        shifted = df.groupby(group_col)[base_cols].shift(lag)

        # 不再 bfill，也不 fillna。
        # 轨迹开头不足 history_len 的样本会保留 NaN，
        # 后续 finite_mask 会把它们过滤掉。
        shifted.columns = [f"hist_{lag:02d}_{c}" for c in base_cols]

        parts.append(shifted)

    history_df = pd.concat(parts, axis=1)
    return history_df


def build_input_df(df: pd.DataFrame, input_cols: list[str]) -> pd.DataFrame:
    """
    兼容两类模型：

    1. 旧单帧模型：
        原始 CSV 列 + previous_action

    2. 历史模型：
        hist_XX_xxx 历史列 + 可选 previous_action
    """
    action_cols = make_cols("actions", 20)
    prev_action_cols = make_cols("previous_action", 20)

    is_history, history_len, base_cols = parse_history_cols(input_cols)

    df = df.copy().reset_index(drop=True)
    prev_actions = build_previous_actions(df, action_cols)
    prev_actions.columns = prev_action_cols
    prev_actions = prev_actions.reset_index(drop=True)

    if is_history:
        df_with_traj = ensure_global_traj_id_for_test(df)
        history_df = build_history_stacked_features(
            df=df_with_traj,
            base_cols=base_cols,
            group_col="global_traj_id",
            history_len=history_len,
        ).reset_index(drop=True)

        feature_df = pd.concat(
            [
                df.reset_index(drop=True),
                history_df,
                prev_actions,
            ],
            axis=1,
        )

        print("[INFO] detected history-stacked input")
        print("[INFO] inferred history_len:", history_len)
        print("[INFO] history base dim:", len(base_cols))
        print("[INFO] history input dim without optional previous_action:", history_len * len(base_cols))
    else:
        feature_df = pd.concat(
            [
                df.reset_index(drop=True),
                prev_actions,
            ],
            axis=1,
        )

        print("[INFO] detected single-frame input")

    missing = [c for c in input_cols if c not in feature_df.columns]
    if missing:
        raise KeyError(
            "模型需要这些输入列，但 CSV/构造特征中找不到:\n"
            f"{missing}\n\n"
            "常见原因:\n"
            "1. policy_name / target_mode 指错，metadata 与模型不匹配；\n"
            "2. 训练用了 full 视觉模型，但 CSV 没有 ball_center；\n"
            "3. 历史模型的 history_len / input_order 与当前测试脚本构造不一致；\n"
            "4. CSV 列名和训练时不一致。"
        )

    return feature_df[input_cols]


# ============================================================
# Inference / Test
# ============================================================

def infer_policy(policy, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            x_t = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            y_t = policy(x_t)
            preds.append(y_t.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def test_one_csv(
    csv_path: Path,
    save_path: Path,
    policy,
    input_cols: list[str],
    target_mode: str,
    args,
    device: torch.device,
):
    print("")
    print("============================================================")
    print("[TEST] csv:", csv_path)
    print("[TEST] save:", save_path)
    print("============================================================")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    df = read_csv_auto(csv_path)
    print("[INFO] rows:", len(df))
    print("[INFO] cols:", len(df.columns))

    input_df = build_input_df(df, input_cols)
    target_df = build_targets(df, target_mode)

    x_all = input_df.to_numpy(dtype=np.float32)
    y_all = target_df.to_numpy(dtype=np.float32)

    finite_mask = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all).all(axis=1)
    valid_indices = np.where(finite_mask)[0]

    if len(valid_indices) == 0:
        raise RuntimeError(f"{csv_path} 没有有效样本，可能存在 NaN/Inf。")

    print("[INFO] valid samples:", len(valid_indices))
    print("[INFO] input dim:", x_all.shape[1])
    print("[INFO] target dim:", y_all.shape[1])

    if args.num_samples is not None and args.num_samples > 0:
        rng = np.random.default_rng(args.seed)
        num_samples = min(args.num_samples, len(valid_indices))
        chosen_indices = rng.choice(valid_indices, size=num_samples, replace=False)
        chosen_indices = np.sort(chosen_indices)
    else:
        chosen_indices = valid_indices
        num_samples = len(chosen_indices)

    x = x_all[chosen_indices]
    y_true = y_all[chosen_indices]

    y_pred = infer_policy(
        policy=policy,
        x=x,
        device=device,
        batch_size=args.infer_batch_size,
    )

    err = y_pred - y_true
    abs_err = np.abs(err)

    mae_per_sample = abs_err.mean(axis=1)
    max_err_per_sample = abs_err.max(axis=1)

    overall_mae = abs_err.mean()
    overall_max = abs_err.max()
    overall_rmse = np.sqrt(np.mean(err ** 2))

    print("")
    print("========== Overall Error ==========")
    print(f"csv          : {csv_path.name}")
    print(f"target_mode  : {target_mode}")
    print(f"num_samples  : {num_samples}")
    print(f"MAE          : {overall_mae:.8f}")
    print(f"RMSE         : {overall_rmse:.8f}")
    print(f"MAX_ABS_ERR  : {overall_max:.8f}")
    print("===================================")

    result_rows = []
    for n, row_idx in enumerate(chosen_indices):
        file_index = df.loc[row_idx, "file_index"] if "file_index" in df.columns else -1
        step = df.loc[row_idx, "step"] if "step" in df.columns else row_idx

        row = {
            "csv_name": csv_path.name,
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

    result_df = pd.DataFrame(result_rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_path, index=False)
    print("[DONE] saved result:", save_path)

    print_num = min(args.print_samples, len(chosen_indices))
    if print_num > 0:
        print("")
        print(f"========== Print First {print_num} Samples ==========")
        for n in range(print_num):
            row_idx = chosen_indices[n]
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

    summary = {
        "csv": csv_path.name,
        "result": save_path.name,
        "target_mode": target_mode,
        "num_samples": int(num_samples),
        "input_dim": int(x_all.shape[1]),
        "target_dim": int(y_all.shape[1]),
        "mae": float(overall_mae),
        "rmse": float(overall_rmse),
        "max_abs_err": float(overall_max),
    }
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_dir",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/data/csv",
        help="存放 CSV 的目录。",
    )

    parser.add_argument(
        "--log_root",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/logs",
        help="所有策略日志的根目录。",
    )

    parser.add_argument(
        "--obs_mode",
        type=str,
        default="no_vision",
        choices=["full", "no_vision"],
        help="full 使用 ball_center；no_vision 不使用 ball_center。",
    )

    parser.add_argument(
        "--use_history_stack",
        action="store_true",
        help="启用后 policy_name 自动追加 _history，用于定位历史模型目录。输入结构仍优先从 metadata 读取。",
    )

    parser.add_argument(
        "--history_len",
        type=int,
        default=10,
        help="历史长度。测试脚本主要从 metadata/input_order 推断，这里用于 metadata 缺失时兜底。",
    )

    parser.add_argument(
        "--history_include_previous_action",
        action="store_true",
        help="metadata 缺失时兜底使用。正常情况下会从 input_order 自动判断是否包含 previous_action。",
    )

    parser.add_argument(
        "--policy_name",
        type=str,
        default="",
        help=(
            "策略名称。为空时根据 obs_mode 自动设置。"
            "如果使用 --use_history_stack，会自动追加 _history 后缀。"
        ),
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="测试结果保存目录。为空时默认保存到 log_root/policy_name/target_mode。",
    )

    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="TorchScript JIT 策略路径。为空时自动从 metadata 或默认路径推断。",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="",
        help="metadata.json 路径。为空时默认使用 log_root/policy_name/target_mode/metadata.json。",
    )

    parser.add_argument(
        "--target_mode",
        type=str,
        default="auto",
        choices=["auto", "actions", "action_delta", "next_actions", "next_joint_delta"],
        help="auto 表示从 metadata 读取；若有多个 target_mode 子目录，建议显式指定。",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="每个 CSV 测试多少条。<=0 表示测试全部有效样本。",
    )

    parser.add_argument(
        "--print_samples",
        type=int,
        default=5,
        help="每个 CSV 只在终端打印前几个样本，避免刷屏。",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--infer_batch_size", type=int, default=4096)

    args = parser.parse_args()

    policy_dir, result_dir, metadata_path, policy_path, metadata, target_mode = resolve_paths(args)

    csv_dir = Path(args.csv_dir)
    input_cols = resolve_input_cols(metadata, args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, use CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("[INFO] obs_mode          :", args.obs_mode)
    print("[INFO] use_history_stack :", args.use_history_stack)
    print("[INFO] history_len       :", args.history_len)
    print("[INFO] policy_name       :", args.policy_name)
    print("[INFO] policy_dir        :", policy_dir)
    print("[INFO] target_mode       :", target_mode)
    print("[INFO] result_dir        :", result_dir)
    print("[INFO] metadata          :", metadata_path)
    print("[INFO] policy            :", policy_path)
    print("[INFO] device            :", device)

    if not Path(args.policy).exists():
        raise FileNotFoundError(
            f"JIT policy 不存在: {args.policy}\n"
            "请检查：\n"
            "1. --policy_name 是否正确；\n"
            "2. --target_mode 是否和训练时一致；\n"
            "3. 是否需要添加 --use_history_stack 来自动定位 _history 目录；\n"
            "4. 训练脚本是否已成功导出 supervised_action_policy_jit.pt。"
        )

    policy = torch.jit.load(args.policy, map_location=device)
    policy.eval()

    test_items = [
        ("replay_data_0615_30HZ_1.csv", "test_result_0615_1.csv"),
        ("replay_data_0528_1.csv", "test_result_0528_1.csv"),
        ("replay_data_0528_1_v2.csv", "test_result_0528_1_v2.csv"),
    ]

    summaries = []
    for csv_name, result_name in test_items:
        csv_path = csv_dir / csv_name
        result_name_with_mode = add_target_mode_suffix(result_name, target_mode)
        save_path = result_dir / result_name_with_mode

        summary = test_one_csv(
            csv_path=csv_path,
            save_path=save_path,
            policy=policy,
            input_cols=input_cols,
            target_mode=target_mode,
            args=args,
            device=device,
        )
        summaries.append(summary)

    summary_path = result_dir / f"test_result_summary_{target_mode}.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print("")
    print("============================================================")
    print("[DONE] all tests finished")
    print("[DONE] summary saved:", summary_path)
    print("============================================================")
    print("")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
