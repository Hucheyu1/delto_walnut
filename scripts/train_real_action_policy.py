from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass
class DatasetInfo:
    input_dim: int
    output_dim: int
    num_samples: int
    input_fields: list[str]
    target_mode: str
    prev_action_mode: str


class ActionDeltaMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 20, hidden_dims: tuple[int, ...] = (256, 256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NormalizedActionPolicy(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / self.input_std
        y_norm = self.model(x_norm)
        return y_norm * self.target_std + self.target_mean


def _to_2d_array(value: Any, key: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] == 0:
        raise ValueError(f"Field '{key}' is empty.")
    return arr


def _load_record(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            obj = pickle.load(f)
    elif path.suffix == ".npz":
        obj = dict(np.load(path))
    else:
        raise ValueError(f"Unsupported data file: {path}. Use .pkl or .npz.")

    if isinstance(obj, list):
        merged: dict[str, list[Any]] = {}
        for frame in obj:
            if not isinstance(frame, dict):
                raise ValueError(f"List data in {path} must contain dict frames.")
            for key, value in frame.items():
                merged.setdefault(key, []).append(value)
        obj = merged
    if not isinstance(obj, dict):
        raise ValueError(f"Data file {path} must contain a dict or list[dict], got {type(obj)}.")

    return {key: _to_2d_array(value, key) for key, value in obj.items()}


def _field(record: dict[str, np.ndarray], key: str, required: bool = True) -> np.ndarray | None:
    if key in record:
        return record[key]
    if required:
        raise KeyError(f"Required field '{key}' not found. Available fields: {list(record.keys())}")
    return None


def _first_existing(record: dict[str, np.ndarray], keys: list[str]) -> tuple[str, np.ndarray]:
    for key in keys:
        if key in record:
            return key, record[key]
    raise KeyError(f"None of these fields exists: {keys}. Available fields: {list(record.keys())}")


def _prev_action(record: dict[str, np.ndarray], args: argparse.Namespace, length: int) -> tuple[str, np.ndarray]:
    explicit = _field(record, args.prev_action_key, required=False)
    if explicit is not None:
        prev = explicit[:length]
        return args.prev_action_key, prev

    if args.prev_action_mode == "actions":
        actions = _field(record, args.action_key)[:length]
        prev = np.zeros_like(actions)
        prev[1:] = actions[:-1]
        return f"prev({args.action_key})", prev

    target_pos = _field(record, args.target_pos_key)
    delta = np.diff(target_pos, axis=0)
    zero = np.zeros_like(delta[:1])
    prev = np.concatenate([zero, delta], axis=0)[:length]
    return f"prev(delta({args.target_pos_key}))", prev


def _target(record: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, int]:
    if args.target_mode == "delta_target_pos":
        target_pos = _field(record, args.target_pos_key)
        return np.diff(target_pos, axis=0), target_pos.shape[0] - 1
    if args.target_mode == "actions":
        actions = _field(record, args.action_key)
        return actions, actions.shape[0]
    if args.target_mode == "next_actions":
        actions = _field(record, args.action_key)
        return actions[1:], actions.shape[0] - 1
    raise ValueError(f"Unknown target mode: {args.target_mode}")


def _build_samples_from_record(record: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target, length = _target(record, args)
    if length <= 1:
        raise ValueError("Record is too short after target construction.")

    fields: list[tuple[str, np.ndarray]] = []

    joint = _field(record, args.joint_key)[:length]
    fields.append((args.joint_key, joint))

    ball1_key, ball1 = _first_existing(record, args.ball1_keys)
    ball2_key, ball2 = _first_existing(record, args.ball2_keys)
    fields.append((ball1_key, ball1[:length]))
    fields.append((ball2_key, ball2[:length]))

    for tactile_key in args.tactile_keys:
        tactile = _field(record, tactile_key)
        fields.append((tactile_key, tactile[:length]))

    for extra_key in args.extra_keys:
        extra = _field(record, extra_key)
        fields.append((extra_key, extra[:length]))

    prev_name, prev = _prev_action(record, args, length)
    fields.append((prev_name, prev))

    min_len = min([arr.shape[0] for _, arr in fields] + [target.shape[0]])
    x = np.concatenate([arr[:min_len] for _, arr in fields], axis=-1)
    y = target[:min_len]

    finite = np.isfinite(x).all(axis=-1) & np.isfinite(y).all(axis=-1)
    x = x[finite]
    y = y[finite]
    if x.shape[0] == 0:
        raise ValueError("No finite samples were produced from this record.")
    return x, y, [name for name, _ in fields]


def load_dataset(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
    files: list[Path] = []
    for item in args.data:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.pkl")))
            files.extend(sorted(path.glob("*.npz")))
        else:
            files.append(path)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No .pkl/.npz data files found.")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    input_fields: list[str] | None = None

    for path in files:
        record = _load_record(path)
        x, y, fields = _build_samples_from_record(record, args)
        xs.append(x)
        ys.append(y)
        if input_fields is None:
            input_fields = fields
        elif fields != input_fields:
            raise ValueError(f"Input fields differ in {path}: {fields} != {input_fields}")
        print(f"[DATA] {path}: {x.shape[0]} samples, input_dim={x.shape[1]}, output_dim={y.shape[1]}")

    x_all = np.concatenate(xs, axis=0).astype(np.float32)
    y_all = np.concatenate(ys, axis=0).astype(np.float32)
    info = DatasetInfo(
        input_dim=x_all.shape[1],
        output_dim=y_all.shape[1],
        num_samples=x_all.shape[0],
        input_fields=input_fields or [],
        target_mode=args.target_mode,
        prev_action_mode=args.prev_action_mode,
    )
    return torch.from_numpy(x_all), torch.from_numpy(y_all), info


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, info = load_dataset(args)
    input_mean = x.mean(dim=0)
    input_std = x.std(dim=0).clamp_min(args.norm_eps)
    target_mean = y.mean(dim=0)
    target_std = y.std(dim=0).clamp_min(args.norm_eps)

    x_norm = (x - input_mean) / input_std
    y_norm = (y - target_mean) / target_std

    dataset = TensorDataset(x_norm, y_norm)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    hidden_dims = tuple(int(dim) for dim in args.hidden_dims)
    model = ActionDeltaMLP(info.input_dim, info.output_dim, hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += loss.item() * batch_x.shape[0]
        train_loss /= max(1, train_size)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_loss += loss_fn(pred, batch_y).item() * batch_x.shape[0]
                pred_raw = pred.cpu() * target_std + target_mean
                target_raw = batch_y.cpu() * target_std + target_mean
                val_mae += torch.mean(torch.abs(pred_raw - target_raw)).item() * batch_x.shape[0]
        val_loss /= max(1, val_size)
        val_mae /= max(1, val_size)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"[TRAIN] epoch={epoch:04d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} val_mae_raw={val_mae:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model_cpu = model.cpu().eval()
    deploy_policy = NormalizedActionPolicy(model_cpu, input_mean, input_std, target_mean, target_std).eval()

    checkpoint = {
        "model_state_dict": model_cpu.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "dataset_info": asdict(info),
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / "real_action_policy.ckpt")

    scripted = torch.jit.script(deploy_policy)
    scripted.save(output_dir / "real_action_policy_jit.pt")

    metadata = {
        "dataset_info": asdict(info),
        "input_order": info.input_fields,
        "target": args.target_mode,
        "checkpoint": str(output_dir / "real_action_policy.ckpt"),
        "jit": str(output_dir / "real_action_policy_jit.pt"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Saved checkpoint to {output_dir / 'real_action_policy.ckpt'}")
    print(f"[DONE] Saved deployable TorchScript policy to {output_dir / 'real_action_policy_jit.pt'}")
    print(f"[INFO] Input fields: {info.input_fields}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised real-data policy that maps tactile/YOLO/joint/previous-action inputs to action deltas."
    )
    parser.add_argument("--data", nargs="+", default=["data"], help="Data files or directories containing .pkl/.npz.")
    parser.add_argument("--output_dir", default="logs/real_action_policy", help="Directory to save the trained model.")
    parser.add_argument("--joint_key", default="obs_joint_pos", help="Field name for finger joint angles.")
    parser.add_argument(
        "--ball1_keys",
        nargs="+",
        default=["ball1_uv", "ball1_pos", "ball1_yolo", "ball1"],
        help="Candidate field names for ball 1 YOLO coordinate.",
    )
    parser.add_argument(
        "--ball2_keys",
        nargs="+",
        default=["ball2_uv", "ball2_pos", "ball2_yolo", "ball2"],
        help="Candidate field names for ball 2 YOLO coordinate.",
    )
    parser.add_argument("--tactile_keys", nargs="*", default=[], help="Optional tactile field names to concatenate.")
    parser.add_argument("--extra_keys", nargs="*", default=[], help="Optional extra input field names to concatenate.")
    parser.add_argument("--prev_action_key", default="prev_action", help="Use this field if it exists.")
    parser.add_argument(
        "--prev_action_mode",
        choices=["actions", "delta_target_pos"],
        default="actions",
        help="How to synthesize previous action if prev_action_key is missing.",
    )
    parser.add_argument("--action_key", default="actions", help="Field name for recorded actions.")
    parser.add_argument("--target_pos_key", default="target_pos", help="Field name for target joint positions.")
    parser.add_argument(
        "--target_mode",
        choices=["delta_target_pos", "actions", "next_actions"],
        default="delta_target_pos",
        help="Training target. delta_target_pos predicts required joint-angle increment.",
    )
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--huber_beta", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--norm_eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
