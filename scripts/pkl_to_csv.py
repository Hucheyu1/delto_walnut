#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def to_numpy(x: Any):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def frame_to_flat_array(x: Any) -> np.ndarray:
    """
    把单帧数据转成一维数组。
    兼容 list / numpy / torch tensor / scalar。
    """
    x = to_numpy(x)

    try:
        arr = np.asarray(x, dtype=np.float32)
    except Exception:
        arr = np.asarray(x, dtype=object)

    if arr.dtype == object:
        flat = []
        for item in arr.reshape(-1):
            item = to_numpy(item)
            try:
                item_arr = np.asarray(item, dtype=np.float32).reshape(-1)
                flat.extend(item_arr.tolist())
            except Exception:
                try:
                    flat.append(float(item))
                except Exception:
                    flat.append(np.nan)
        return np.asarray(flat, dtype=np.float32)

    return arr.reshape(-1).astype(np.float32)


def sequence_to_2d_array(value: Any, key: str) -> np.ndarray:
    """
    保留第一维为时间维 T，把后面全部展平成特征维 D。

    重点兼容 actions 这种不规整 list：
        actions = [array(...), array(...), ...]
    """
    value = to_numpy(value)

    # 先尝试直接规整转换
    try:
        arr = np.asarray(value, dtype=np.float32)

        if arr.ndim == 0:
            return arr.reshape(1, 1)

        if arr.ndim == 1:
            # 如果是一维数值数组，例如 duration=[12.3]，返回 [1, 1] 或 [T, 1]
            return arr.reshape(-1, 1)

        return arr.reshape(arr.shape[0], -1)

    except Exception:
        pass

    # 如果直接转换失败，说明可能是不规整 list，逐帧处理
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} cannot be converted to array, type={type(value)}")

    frames = [frame_to_flat_array(v) for v in value]

    if len(frames) == 0:
        raise ValueError(f"{key} is empty")

    max_dim = max(f.size for f in frames)

    arr = np.full((len(frames), max_dim), np.nan, dtype=np.float32)

    for i, f in enumerate(frames):
        arr[i, : f.size] = f

    return arr


def infer_time_length(arrays: dict[str, np.ndarray]) -> int:
    """
    推断主时间长度 T。
    忽略 duration 这种只有 1 行的元信息。
    """
    lengths = []

    for key, arr in arrays.items():
        if arr.shape[0] > 1:
            lengths.append(arr.shape[0])

    if not lengths:
        return 1

    # 取出现最多的长度；你的情况应该是 752
    unique, counts = np.unique(lengths, return_counts=True)
    T = int(unique[np.argmax(counts)])
    return T


def pkl_to_csv(input_pkl: str, output_csv: str | None = None):
    input_pkl = Path(input_pkl)

    if output_csv is None:
        output_csv = input_pkl.with_suffix(".csv")
    else:
        output_csv = Path(output_csv)

    with input_pkl.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in pkl, got {type(data)}")

    print("[INFO] loaded:", input_pkl)
    print("[INFO] keys:", list(data.keys()))

    arrays = {}

    for key, value in data.items():
        try:
            arr = sequence_to_2d_array(value, key)
            arrays[key] = arr
            print(f"[FIELD] {key}: shape={arr.shape}")
        except Exception as e:
            print(f"[WARN] skip {key}: {e}")

    if not arrays:
        raise RuntimeError("No valid fields found.")

    T = infer_time_length(arrays)
    print("[INFO] inferred T:", T)

    rows = {
        "step": np.arange(T, dtype=np.int32)
    }

    for key, arr in arrays.items():
        # 如果是 duration 这种只有 1 行的元信息，重复 T 次
        if arr.shape[0] == 1 and T > 1:
            arr = np.repeat(arr, T, axis=0)

        # 如果某个字段长度比 T 短，补 NaN
        if arr.shape[0] < T:
            pad = np.full((T - arr.shape[0], arr.shape[1]), np.nan, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        # 如果某个字段长度比 T 长，截断
        if arr.shape[0] > T:
            arr = arr[:T]

        dim = arr.shape[1]

        if dim == 1:
            rows[key] = arr[:, 0]
        else:
            for i in range(dim):
                rows[f"{key}_{i:02d}"] = arr[:, i]

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("[DONE] saved:", output_csv)
    print("[DONE] rows:", len(df))
    print("[DONE] cols:", len(df.columns))

    print("\n[INFO] first columns:")
    for col in df.columns[:50]:
        print(" ", col)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pkl", type=str)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    pkl_to_csv(args.input_pkl, args.output_csv)


if __name__ == "__main__":
    main()
