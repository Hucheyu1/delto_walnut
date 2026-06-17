#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def natural_key(path: Path):
    """
    保证 data_2.pkl 排在 data_10.pkl 前面。
    """
    numbers = re.findall(r"\d+", path.stem)
    if numbers:
        return int(numbers[-1])
    return path.stem


def torch_to_cpu(x: Any):
    """
    只处理 torch.Tensor。
    不在这里强制 np.asarray，因为 actions 可能是不规整 list。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def frame_to_flat_array(x: Any) -> np.ndarray:
    """
    把单帧数据转成一维 float32 数组。

    兼容：
        tensor([20])
        tensor([1, 20])
        list[float]
        nested list
        scalar
    """
    x = torch_to_cpu(x)

    try:
        arr = np.asarray(x, dtype=np.float32)
        return arr.reshape(-1).astype(np.float32)
    except Exception:
        pass

    # 如果还是不行，说明里面可能还有 tensor/list 混杂，逐个拆
    flat = []

    try:
        obj_arr = np.asarray(x, dtype=object).reshape(-1)
    except Exception:
        obj_arr = [x]

    for item in obj_arr:
        item = torch_to_cpu(item)

        try:
            item_arr = np.asarray(item, dtype=np.float32).reshape(-1)
            flat.extend(item_arr.tolist())
        except Exception:
            try:
                flat.append(float(item))
            except Exception:
                flat.append(np.nan)

    return np.asarray(flat, dtype=np.float32)


def to_2d_time_array(value: Any, key: str) -> np.ndarray:
    """
    把字段统一整理成 [T, D]。

    重点：
        - 保留第 0 维作为时间维 T
        - actions 这种不规整 list 会逐帧展开
        - duration 这种单个值会变成 [1, 1]
    """
    value = torch_to_cpu(value)

    # 第一优先：直接转成规整 float 数组
    try:
        arr = np.asarray(value, dtype=np.float32)

        if arr.ndim == 0:
            arr = arr.reshape(1, 1)

        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        else:
            arr = arr.reshape(arr.shape[0], -1)

        if arr.shape[0] == 0:
            raise ValueError(f"Field {key} is empty.")

        return arr.astype(np.float32)

    except Exception:
        pass

    # 第二优先：处理 actions 这种 list[tensor/list]，逐帧 flatten
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Field {key} cannot be converted, type={type(value)}")

    frames = [frame_to_flat_array(v) for v in value]

    if len(frames) == 0:
        raise ValueError(f"Field {key} is empty.")

    max_dim = max(f.size for f in frames)

    arr = np.full((len(frames), max_dim), np.nan, dtype=np.float32)

    for i, f in enumerate(frames):
        arr[i, : f.size] = f

    return arr


def infer_main_length(arrays: dict[str, np.ndarray]) -> int:
    """
    推断主时间长度 T。

    不能用 min_len，因为 duration 只有 1 行。
    应该忽略长度为 1 的元信息字段，然后取出现最多的长度。

    你的情况：
        ball_center            752
        tactile_data           752
        tesollo_joints_state   752
        actions                752
        duration               1

    推断出来 T = 752。
    """
    lengths = []

    for key, arr in arrays.items():
        if arr.shape[0] > 1:
            lengths.append(arr.shape[0])

    if len(lengths) == 0:
        return 1

    unique, counts = np.unique(lengths, return_counts=True)
    main_len = int(unique[np.argmax(counts)])
    return main_len


def align_array_to_length(arr: np.ndarray, T: int) -> np.ndarray:
    """
    把字段对齐到主长度 T。

    - 如果 arr 只有 1 行，比如 duration，就重复 T 次
    - 如果 arr 短于 T，就补 NaN
    - 如果 arr 长于 T，就截断
    """
    if arr.shape[0] == T:
        return arr

    if arr.shape[0] == 1 and T > 1:
        return np.repeat(arr, T, axis=0)

    if arr.shape[0] < T:
        pad = np.full((T - arr.shape[0], arr.shape[1]), np.nan, dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    return arr[:T]


def flatten_frame_dict(frame: dict[str, Any]) -> dict[str, Any]:
    """
    如果 pkl 是 list[dict] 格式，用这个函数展开每一帧。
    """
    row = {}

    for key, value in frame.items():
        arr = frame_to_flat_array(value)

        if arr.size == 1:
            row[key] = arr[0].item()
        else:
            for i, v in enumerate(arr):
                row[f"{key}_{i:02d}"] = v.item() if hasattr(v, "item") else v

    return row


def record_dict_to_dataframe(record: dict[str, Any], source_file: str, file_index: int) -> pd.DataFrame:
    """
    处理这种格式：

    data = {
        "ball_center":          shape=(752, 4),
        "tactile_data":         shape=(752, 13),
        "tesollo_joints_state": shape=(752, 20),
        "actions":              shape=(752, ?),
        "duration":             shape=(1, 1),
    }

    输出：
        每一帧一行，共 752 行。
    """
    arrays: dict[str, np.ndarray] = {}

    for key, value in record.items():
        try:
            arr = to_2d_time_array(value, key)
            arrays[key] = arr
            print(f"       [FIELD] {key}: shape={arr.shape}")
        except Exception as e:
            print(f"       [WARN] skip key={key}, reason={e}")

    if not arrays:
        raise ValueError(f"No valid fields in {source_file}")

    T = infer_main_length(arrays)

    print(f"       [INFO] inferred T={T}")

    rows = {
        "file_index": np.full(T, file_index, dtype=np.int32),
        "source_file": np.full(T, source_file),
        "step": np.arange(T, dtype=np.int32),
    }

    for key, arr in arrays.items():
        arr = align_array_to_length(arr, T)

        dim = arr.shape[1]

        if dim == 1:
            rows[key] = arr[:, 0]
        else:
            for j in range(dim):
                rows[f"{key}_{j:02d}"] = arr[:, j]

    df = pd.DataFrame(rows)
    return df


def record_list_to_dataframe(record: list[dict[str, Any]], source_file: str, file_index: int) -> pd.DataFrame:
    """
    处理这种格式：

    data = [
        {"actions": ..., "obs_joint_pos": ...},
        {"actions": ..., "obs_joint_pos": ...},
        ...
    ]
    """
    rows = []

    for step, frame in enumerate(record):
        if not isinstance(frame, dict):
            raise ValueError(f"list record must contain dict frames, got {type(frame)}")

        row = {
            "file_index": file_index,
            "source_file": source_file,
            "step": step,
        }

        row.update(flatten_frame_dict(frame))
        rows.append(row)

    return pd.DataFrame(rows)


def load_one_pkl(path: Path, file_index: int) -> pd.DataFrame:
    """
    加载单个 pkl 并转成 DataFrame。
    """
    with path.open("rb") as f:
        record = pickle.load(f)

    source_file = path.name

    if isinstance(record, dict):
        df = record_dict_to_dataframe(record, source_file, file_index)

    elif isinstance(record, list):
        df = record_list_to_dataframe(record, source_file, file_index)

    else:
        raise ValueError(f"Unsupported pkl format in {path}: {type(record)}")

    return df


def merge_pkls(input_dir: Path, output_csv: Path, start: int = 0, end: int = 128):
    """
    合并 data_0.pkl 到 data_128.pkl。
    """
    all_files = sorted(input_dir.glob("data_*.pkl"), key=natural_key)

    selected_files = []
    for path in all_files:
        numbers = re.findall(r"\d+", path.stem)
        if not numbers:
            continue

        idx = int(numbers[-1])
        if start <= idx <= end:
            selected_files.append(path)

    if not selected_files:
        raise FileNotFoundError(f"No data_*.pkl found in {input_dir}")

    print(f"[INFO] input_dir: {input_dir}")
    print(f"[INFO] output_csv: {output_csv}")
    print(f"[INFO] found {len(selected_files)} files")

    dfs = []

    for path in selected_files:
        file_index = natural_key(path)
        print(f"[LOAD] {path.name}")

        try:
            df = load_one_pkl(path, file_index)
        except Exception as e:
            print(f"[ERROR] failed to load {path}: {e}")
            continue

        print(f"       rows={len(df)}, cols={len(df.columns)}")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid pkl files were loaded.")

    merged = pd.concat(dfs, axis=0, ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("")
    print("[DONE] merged csv saved:")
    print(f"       {output_csv}")
    print(f"[DONE] total rows: {len(merged)}")
    print(f"[DONE] total cols: {len(merged.columns)}")
    print("")
    print("[INFO] first 80 columns:")
    for col in list(merged.columns)[:80]:
        print(f"  {col}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/data/replay_data_0615_30HZ_1",
        help="包含 data_0.pkl 到 data_128.pkl 的文件夹",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/data/replay_data_0615_30HZ_1/replay_data_0615_30HZ_1.csv",
        help="输出 CSV 路径",
    )

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=128)

    args = parser.parse_args()

    merge_pkls(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
