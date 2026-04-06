#!/usr/bin/env python3
"""Run residue-identity inference with a trained voxel CNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.cnn3d import CNN3DConfig, VoxelCNN3D
from protein_constants import LABEL_RES_DICT, LETTER1_3_DICT, RES_LABEL_DICT


class PredictionManifestDataset(Dataset):
    """Manifest dataset compatible with train_voxel_cnn.py conventions."""

    def __init__(self, manifest_path: Path, mean: np.ndarray, std: np.ndarray, task: str):
        self.manifest_path = manifest_path
        self.manifest_dir = manifest_path.parent.resolve()
        self.task = task
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        first = self.rows[0]
        if "sample_path" in first:
            self.path_key = "sample_path"
        elif "path" in first:
            self.path_key = "path"
        else:
            raise ValueError("Manifest must contain 'sample_path' or 'path'")

        self.has_sample_index = "sample_index" in first
        self.schema = "sharded" if self.has_sample_index else "per_example"
        self._shard_cache: dict[Path, np.ndarray | np.lib.npyio.NpzFile] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.manifest_dir / path

    @staticmethod
    def _to_channels_first(sample: np.ndarray) -> np.ndarray:
        if sample.ndim == 4 and sample.shape[-1] <= 8 and sample.shape[0] > 8:
            return np.transpose(sample, (3, 0, 1, 2))
        return sample

    @staticmethod
    def _maybe_parse_label(label_value: Any) -> int | None:
        if label_value is None:
            return None
        text = str(label_value).strip()
        if text == "":
            return None

        try:
            return int(text)
        except ValueError:
            upper_text = text.upper()
            if upper_text in RES_LABEL_DICT:
                return RES_LABEL_DICT[upper_text]
            if len(upper_text) == 1 and upper_text in LETTER1_3_DICT:
                return RES_LABEL_DICT[LETTER1_3_DICT[upper_text]]
            return None

    def _load_shard(self, path: Path) -> np.ndarray | np.lib.npyio.NpzFile:
        if path not in self._shard_cache:
            # Intentional shard caching (same behavior class as training dataset).
            self._shard_cache[path] = np.load(path, allow_pickle=False)
        return self._shard_cache[path]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        raw_sample_path = row[self.path_key]
        sample_path = self._resolve_path(raw_sample_path)

        manifest_label = self._maybe_parse_label(row.get("label"))
        npz_label: int | None = None

        if self.schema == "sharded" and row.get("sample_index", "").strip() != "":
            sample_index = int(row["sample_index"])
            sample_obj = self._load_shard(sample_path)
            sample = sample_obj[sample_index].astype(np.float32)
        else:
            with np.load(sample_path, allow_pickle=False) as sample_obj:
                if "x" not in sample_obj:
                    raise ValueError(f"Per-example file missing 'x': {sample_path}")
                sample = sample_obj["x"].astype(np.float32)
                if "y" in sample_obj:
                    npz_label = int(np.asarray(sample_obj["y"]).item())

        if manifest_label is not None and npz_label is not None and manifest_label != npz_label:
            raise ValueError(
                f"Label mismatch for {raw_sample_path}: manifest={manifest_label} npz={npz_label}"
            )

        sample = (sample - self.mean) / self.std
        sample = self._to_channels_first(sample)

        if sample.ndim == 3:
            sample = sample[np.newaxis, ...]
        elif sample.ndim != 4:
            raise ValueError(f"Expected 3D or 4D sample, got shape {sample.shape} for {raw_sample_path}")

        actual_idx = manifest_label if manifest_label is not None else npz_label

        return {
            "x": torch.from_numpy(sample),
            "actual_idx": actual_idx,
            "row": row,
            "raw_sample_path": raw_sample_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict residue identity from voxel manifest")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)

    parser.add_argument("--task", choices=["residue_identity"], default="residue_identity")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--output-probs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(path)
    if "train_mean" not in stats or "train_std" not in stats:
        raise ValueError("Normalization file must contain train_mean and train_std")
    return stats["train_mean"], stats["train_std"]


def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int | None,
    task: str,
) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing model_state")

    ckpt_config = checkpoint.get("config")
    if ckpt_config is None:
        if num_classes is None:
            raise ValueError("--num-classes is required when checkpoint config is unavailable")
        out_features = checkpoint["model_state"]["head.3.weight"].shape[0]
        resolved_classes = num_classes if num_classes is not None else out_features
        config = CNN3DConfig(in_channels=1, num_classes=resolved_classes, task=task)
    else:
        config_dict = dict(ckpt_config)
        config_dict["task"] = task
        if num_classes is not None:
            config_dict["num_classes"] = num_classes
        config = CNN3DConfig(**config_dict)

    model = VoxelCNN3D(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, int(config.num_classes)


def idx_to_residue_map() -> dict[int, str]:
    return dict(LABEL_RES_DICT)


def label_to_index(label: str) -> int | None:
    upper_label = label.strip().upper()
    if upper_label in RES_LABEL_DICT:
        return RES_LABEL_DICT[upper_label]
    if len(upper_label) == 1 and upper_label in LETTER1_3_DICT:
        return RES_LABEL_DICT[LETTER1_3_DICT[upper_label]]
    return None


def normalize_residue_label(value: Any, idx_map: dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return idx_map.get(int(value), str(int(value)))

    text = str(value).strip()
    if text == "":
        return ""

    maybe_int = PredictionManifestDataset._maybe_parse_label(text)
    if maybe_int is not None:
        return idx_map.get(int(maybe_int), str(maybe_int))

    upper_text = text.upper()
    if len(upper_text) == 1 and upper_text in LETTER1_3_DICT:
        return LETTER1_3_DICT[upper_text]
    return upper_text


def format_site(row: dict[str, str], sample_path: Path) -> str:
    chain_id = (row.get("chain_id") or "").strip()
    res_no = (row.get("res_no") or "").strip()
    res_name = (row.get("res_name") or "").strip()

    if chain_id and res_no and res_name:
        return f"{chain_id}:{res_no}:{res_name.upper()}"

    example_id = (row.get("example_id") or "").strip()
    if example_id:
        return example_id

    stem = sample_path.stem
    # Very common fallback pattern: <structure>__<chain>:<res_no>:<res_name>
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def infer_structure_name(row: dict[str, str], sample_path: Path) -> str:
    structure_id = (row.get("structure_id") or "").strip()
    if structure_id:
        return structure_id

    stem = sample_path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]

    parent_name = sample_path.parent.name.strip()
    if parent_name:
        return parent_name

    return stem


def collate_prediction_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    xs = torch.stack([item["x"] for item in batch], dim=0)
    return {
        "x": xs,
        "actual_idx": [item["actual_idx"] for item in batch],
        "rows": [item["row"] for item in batch],
        "sample_paths": [item["raw_sample_path"] for item in batch],
    }


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_map: dict[int, str],
    top_k: int,
    output_probs: bool,
    amp_enabled: bool,
    verbose: bool,
) -> tuple[list[dict[str, str]], float, float | None]:
    rows_out: list[dict[str, str]] = []

    correct_top1 = 0
    correct_topk = 0
    with_actual = 0

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting", unit="batch"):
            x = batch["x"].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(x)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            k = max(1, min(top_k, probs.shape[1]))
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)

            pred_indices = preds.detach().cpu().numpy().astype(int)
            topk_indices_np = topk_indices.detach().cpu().numpy().astype(int)
            topk_probs_np = topk_probs.detach().cpu().numpy().astype(np.float32)

            for i, pred_idx in enumerate(pred_indices):
                row = batch["rows"][i]
                sample_path = Path(batch["sample_paths"][i])

                actual_idx = batch["actual_idx"][i]
                if actual_idx is not None:
                    with_actual += 1
                    correct_top1 += int(pred_idx == actual_idx)
                    correct_topk += int(actual_idx in topk_indices_np[i])

                out_row = {
                    "structure_name": infer_structure_name(row, sample_path),
                    "site": format_site(row, sample_path),
                    "predicted": idx_map.get(pred_idx, str(pred_idx)),
                    "actual": normalize_residue_label(actual_idx, idx_map),
                }

                if output_probs:
                    topk_labels = [idx_map.get(int(idx), str(int(idx))) for idx in topk_indices_np[i]]
                    out_row.update(
                        {
                            "predicted_index": str(pred_idx),
                            "actual_index": "" if actual_idx is None else str(int(actual_idx)),
                            "confidence": f"{float(topk_probs_np[i][0]):.6f}",
                            "topk_indices": "|".join(str(int(idx)) for idx in topk_indices_np[i]),
                            "topk_labels": "|".join(topk_labels),
                        }
                    )

                if verbose and actual_idx is None:
                    print(f"No ground truth for sample: {sample_path}")

                rows_out.append(out_row)

    accuracy = (correct_top1 / with_actual) if with_actual > 0 else float("nan")
    topk_accuracy = (correct_topk / with_actual) if with_actual > 0 else None
    return rows_out, accuracy, topk_accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    mean, std = load_normalization(args.normalization)

    dataset = PredictionManifestDataset(
        manifest_path=args.test_manifest,
        mean=mean,
        std=std,
        task=args.task,
    )

    model, num_classes = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=args.num_classes,
        task=args.task,
    )

    idx_map = idx_to_residue_map()
    if len(idx_map) < num_classes:
        raise ValueError(
            f"Residue index map supports {len(idx_map)} classes, but model has {num_classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prediction_batch,
    )

    rows_out, accuracy, topk_accuracy = predict(
        model=model,
        loader=loader,
        device=device,
        idx_map=idx_map,
        top_k=args.top_k,
        output_probs=args.output_probs,
        amp_enabled=args.amp and device.type == "cuda",
        verbose=args.verbose,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["structure_name", "site", "predicted", "actual"]
    if args.output_probs:
        fieldnames.extend(["predicted_index", "actual_index", "confidence", "topk_indices", "topk_labels"])

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Processed samples: {len(rows_out)}")
    if np.isfinite(accuracy):
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("Accuracy: N/A (no ground-truth labels available)")

    if args.top_k > 1:
        if topk_accuracy is not None:
            print(f"Top-{args.top_k} accuracy: {topk_accuracy:.4f}")
        else:
            print(f"Top-{args.top_k} accuracy: N/A (no ground-truth labels available)")


if __name__ == "__main__":
    main()
