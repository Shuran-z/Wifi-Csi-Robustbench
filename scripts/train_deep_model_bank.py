#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from csi_er.data import load_processed
from csi_er.models import build_model, count_parameters
from csi_er.utils import PROJECT_ROOT, now_ts, set_seed
from scripts.run_research_grade_existing_models import CORRUPTION_SEEDS, CORRUPTIONS, SEVERITIES, apply_research_corruption


FAIL_COLUMNS = ["configuration_id", "base_configuration_id", "dataset", "error", "traceback"]
CHECKPOINT_COLUMNS = ["configuration_id", "run_id", "checkpoint_path", "checkpoint_bytes", "param_count"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train/evaluate the preregistered deep WiFi CSI model bank.")
    ap.add_argument("--config", default="configs/deep_model_bank.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dataset", default="UT_HAR")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--corruption-seed-limit", type=int, default=None)
    ap.add_argument("--severity-limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run_rows(cfg: dict, dataset_filter: str | None = None) -> list[dict]:
    rows = []
    for dataset, family, size, seed in itertools.product(cfg["datasets"], cfg["families"].keys(), ["small", "medium", "large"], cfg["train_seeds"]):
        if dataset_filter and dataset != dataset_filter:
            continue
        params = dict(cfg["families"][family][size])
        base = f"deep__{family}__{size}"
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "size": size,
                "train_seed": int(seed),
                "model_params": params,
                "base_configuration_id": base,
                "configuration_id": f"{base}__seed{seed}",
                "run_id": f"{base}__seed{seed}",
            }
        )
    return rows


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size)
    out: list[int] = []
    with torch.no_grad():
        for (xb,) in loader:
            out.extend(model(xb.to(device)).argmax(1).cpu().numpy().tolist())
    return np.asarray(out, dtype=np.int64)


def loaders(x_train: np.ndarray, y_train: np.ndarray, seed: int, batch_size: int, val_fraction: float):
    indices = np.arange(len(y_train))
    tr_idx, va_idx = train_test_split(indices, test_size=val_fraction, random_state=seed, stratify=y_train)
    train_ds = TensorDataset(torch.tensor(x_train[tr_idx], dtype=torch.float32), torch.tensor(y_train[tr_idx], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(x_train[va_idx], dtype=torch.float32), torch.tensor(y_train[va_idx], dtype=torch.long))
    bs = min(int(batch_size), max(4, len(train_ds)))
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()),
    )


def eval_acc(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yp.extend(model(xb.to(device)).argmax(1).cpu().numpy().tolist())
            yt.extend(yb.numpy().tolist())
    return float(accuracy_score(yt, yp)) if yt else 0.0


def train_one(row: dict, cfg: dict, data: dict, out: Path, args: argparse.Namespace, device: torch.device):
    seed = int(row["train_seed"])
    set_seed(seed)
    x_train, y_train = data["X_train"], data["y_train"]
    x_test, y_test = data["X_test"], data["y_test"]
    input_shape = tuple(x_train.shape[1:])
    ncls = int(max(y_train.max(), y_test.max()) + 1)
    train_cfg = cfg["training"]
    epochs = int(args.epochs or (1 if args.smoke else train_cfg.get("epochs", 30)))
    train_loader, val_loader = loaders(x_train, y_train, seed, train_cfg.get("batch_size", 128), train_cfg.get("val_fraction", 0.15))
    model = build_model(row["family"], ncls, input_shape=input_shape, **row["model_params"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    crit = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("use_amp", True) and device.type == "cuda"))
    best = -1.0
    best_state = None
    patience = 0
    curves = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_acc = eval_acc(model, val_loader, device)
        curves.append(
            {
                "timestamp": now_ts(),
                "dataset": row["dataset"],
                "configuration_id": row["configuration_id"],
                "run_id": row["run_id"],
                "family": row["family"],
                "model_size": row["size"],
                "train_seed": seed,
                "epoch": ep,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_acc": val_acc,
            }
        )
        if val_acc > best:
            best = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(train_cfg.get("patience", 8)) and not args.smoke:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict(model, x_test, device, int(train_cfg.get("batch_size", 128)))
    clean = {
        "timestamp": now_ts(),
        "dataset": row["dataset"],
        "model_name": row["configuration_id"],
        "configuration_id": row["configuration_id"],
        "run_id": row["run_id"],
        "base_configuration_id": row["base_configuration_id"],
        "feature_family": "deep",
        "classifier_family": "deep",
        "architecture_family": row["family"],
        "model_group": row["family"],
        "is_classical": False,
        "train_seed": seed,
        "train_fraction": 1.0,
        "model_size": row["size"],
        "param_count": int(count_parameters(model)),
        "train_time_sec": float(time.time() - t0),
        "best_val_acc": float(best),
        "model_params": json.dumps(row["model_params"], sort_keys=True),
    }
    clean.update({f"clean_{k}": v for k, v in metrics(y_test, pred).items()})
    ckpt = out / "checkpoints" / "deep_bank" / f"{row['configuration_id']}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "row": row,
            "num_classes": ncls,
            "input_shape": list(input_shape),
            "best_val_acc": best,
        },
        ckpt,
    )
    return model, clean, curves, {
        "configuration_id": row["configuration_id"],
        "run_id": row["run_id"],
        "checkpoint_path": str(ckpt.relative_to(out)),
        "checkpoint_bytes": ckpt.stat().st_size,
        "param_count": clean["param_count"],
    }


def write_tables(out: Path, clean_rows: list[dict], corr_rows: list[dict], curve_rows: list[dict], fail_rows: list[dict], ckpt_rows: list[dict]) -> None:
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean_rows).to_csv(out / "results" / "deep_bank_clean.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "results" / "deep_bank_corruptions_long.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "results" / "deep_bank_training_curves.csv", index=False)
    pd.DataFrame(fail_rows, columns=FAIL_COLUMNS).to_csv(out / "results" / "deep_bank_failures.csv", index=False)
    pd.DataFrame(ckpt_rows, columns=CHECKPOINT_COLUMNS).to_csv(out / "manifests" / "deep_bank_checkpoint_manifest.csv", index=False)


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    rows = build_run_rows(cfg, args.dataset)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    out = PROJECT_ROOT / (args.output_dir or f"outputs_final_complete_{time.strftime('%Y%m%d_%H%M%S')}")
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "manifests" / "deep_bank_selected_manifest.csv", index=False)
    all_rows = build_run_rows(cfg, None)
    pd.DataFrame(all_rows).to_csv(out / "manifests" / "deep_bank_preregistered_manifest.csv", index=False)
    print(f"planned_deep_runs={len(rows)}")
    for row in rows[:20]:
        print((row["dataset"], row["family"], row["size"], row["train_seed"]))
    if args.dry_run or not args.execute:
        if args.dry_run:
            print("dry-run: no training launched")
        else:
            print("Use --execute to train/evaluate the bank.")
        return 0
    if args.dataset != "UT_HAR":
        raise FileNotFoundError(f"Dataset loader is not available yet for {args.dataset}; use --dataset UT_HAR for Phase B.")
    data = load_processed(smoke=args.smoke)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    corr_seeds = CORRUPTION_SEEDS[: args.corruption_seed_limit] if args.corruption_seed_limit else CORRUPTION_SEEDS
    severities = SEVERITIES[: args.severity_limit] if args.severity_limit else SEVERITIES
    clean_rows: list[dict] = []
    corr_rows: list[dict] = []
    curve_rows: list[dict] = []
    fail_rows: list[dict] = []
    ckpt_rows: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        try:
            model, clean, curves, ckpt = train_one(row, cfg, data, out, args, device)
            clean_rows.append(clean)
            curve_rows.extend(curves)
            ckpt_rows.append(ckpt)
            for corr_seed in corr_seeds:
                for corruption in CORRUPTIONS:
                    for severity in severities:
                        x_corr = apply_research_corruption(data["X_test"], corruption, severity, seed=int(corr_seed))
                        pred = predict(model, x_corr, device, int(cfg["training"].get("batch_size", 128)))
                        m = metrics(data["y_test"], pred)
                        corr_row = {k: clean[k] for k in [
                            "dataset", "model_name", "configuration_id", "run_id", "base_configuration_id", "feature_family",
                            "classifier_family", "architecture_family", "model_group", "is_classical", "train_seed",
                            "train_fraction", "model_size"
                        ]}
                        corr_row.update(m)
                        corr_row.update(
                            {
                                "corruption": corruption,
                                "severity": int(severity),
                                "corruption_seed": int(corr_seed),
                                "condition_id": f"{corruption}__s{severity}__seed{corr_seed}",
                            }
                        )
                        corr_rows.append(corr_row)
            print(f"[{idx}/{len(rows)}] OK {row['configuration_id']} clean_macro_f1={clean['clean_macro_f1']:.4f}", flush=True)
        except Exception as exc:
            fail_rows.append(
                {
                    "configuration_id": row["configuration_id"],
                    "base_configuration_id": row["base_configuration_id"],
                    "dataset": row["dataset"],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[{idx}/{len(rows)}] FAIL {row['configuration_id']}: {exc}", flush=True)
        write_tables(out, clean_rows, corr_rows, curve_rows, fail_rows, ckpt_rows)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "artifacts" / "deep_bank_run_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(out),
                "smoke": bool(args.smoke),
                "dataset": args.dataset,
                "selected_runs": len(rows),
                "completed_runs": len(clean_rows),
                "failed_runs": len(fail_rows),
                "corruption_rows": len(corr_rows),
                "corruption_seeds": [int(x) for x in corr_seeds],
                "severities": [int(x) for x in severities],
                "device": str(device),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 1 if fail_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
