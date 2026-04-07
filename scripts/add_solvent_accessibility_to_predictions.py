#!/usr/bin/env python3
"""Add solvent accessibility metrics to residue-identity predictions and plot SASA/accuracy summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import freesasa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


ManifestKey = Tuple[str, str, str, str]  # (structure_id, chain_id, res_no, res_name_upper)
ResidueKey = Tuple[str, str, str]  # (chain_id, res_no, res_name_upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append SASA fields to prediction CSV and plot binned top-k accuracy."
    )
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--pdb-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--plot-png", required=True, type=Path)
    parser.add_argument("--plot-mode", choices=["line", "grouped_boxplot"], default="grouped_boxplot")

    parser.add_argument("--sasa-kind", choices=["absolute", "relative"], default="relative")
    parser.add_argument(
        "--sasa-field",
        choices=["total", "sidechain", "mainchain", "polar", "apolar"],
        default="total",
    )

    parser.add_argument(
        "--binning",
        choices=["equal_width", "equal_frequency", "custom"],
        default="equal_width",
    )
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--custom-bins", type=str, default=None)

    parser.add_argument("--cache-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--strict-match", action="store_true")

    return parser.parse_args()


def load_predictions_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_site(site_str: str) -> Tuple[str, str, str]:
    parts = (site_str or "").split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid site format '{site_str}', expected <chain>:<res_no>:<res_name>")
    chain_id, res_no, res_name = parts
    return chain_id.strip(), str(res_no).strip(), res_name.strip()


def normalize_residue_3letter(name: str, output_case: str = "upper") -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned[:3]
    if output_case == "upper":
        return cleaned.upper()
    if output_case == "title":
        return cleaned.upper().lower().capitalize()
    raise ValueError(f"Unknown output_case '{output_case}'")


def build_manifest_lookup(rows: Sequence[dict]) -> Tuple[Dict[ManifestKey, dict], Dict[Tuple[str, str, str], List[dict]]]:
    exact: Dict[ManifestKey, dict] = {}
    fallback: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        structure_id = (row.get("structure_id") or "").strip()
        chain_id = (row.get("chain_id") or "").strip()
        res_no = str(row.get("res_no") or "").strip()
        res_name = normalize_residue_3letter(row.get("res_name") or "", output_case="upper")
        key = (structure_id, chain_id, res_no, res_name)
        exact[key] = row
        fallback[(structure_id, chain_id, res_no)].append(row)
    return exact, fallback


def resolve_prediction_row(
    pred_row: dict,
    manifest_lookup: Dict[ManifestKey, dict],
    fallback_lookup: Dict[Tuple[str, str, str], List[dict]],
    strict_match: bool,
) -> Tuple[Optional[dict], str, str, str, str, Optional[str]]:
    structure_name = (pred_row.get("structure_name") or "").strip()
    chain_id, res_no, site_res_name = parse_site(pred_row.get("site") or "")
    site_res_name_upper = normalize_residue_3letter(site_res_name, output_case="upper")

    exact_key: ManifestKey = (structure_name, chain_id, res_no, site_res_name_upper)
    manifest_row = manifest_lookup.get(exact_key)
    if manifest_row is not None:
        return manifest_row, structure_name, chain_id, res_no, site_res_name_upper, None

    if strict_match:
        return None, structure_name, chain_id, res_no, site_res_name_upper, "strict exact match failed"

    candidates = fallback_lookup.get((structure_name, chain_id, res_no), [])
    if len(candidates) == 1:
        m = candidates[0]
        manifest_res_name = normalize_residue_3letter(m.get("res_name") or "", output_case="upper")
        warning = None
        if manifest_res_name != site_res_name_upper:
            warning = (
                f"residue name mismatch at fallback match: site={site_res_name_upper}, manifest={manifest_res_name}"
            )
        return m, structure_name, chain_id, res_no, manifest_res_name, warning

    if len(candidates) > 1:
        return None, structure_name, chain_id, res_no, site_res_name_upper, "ambiguous fallback match"

    return None, structure_name, chain_id, res_no, site_res_name_upper, "no manifest match"


def pdb_path_for_structure(structure_id: str, pdb_root: Path) -> Path:
    return pdb_root / f"{structure_id}.pdb"


def select_sasa_value(residue_area, sasa_kind: str, sasa_field: str) -> float:
    mapping = {
        ("absolute", "total"): "total",
        ("absolute", "sidechain"): "sideChain",
        ("absolute", "mainchain"): "mainChain",
        ("absolute", "polar"): "polar",
        ("absolute", "apolar"): "apolar",
        ("relative", "total"): "relativeTotal",
        ("relative", "sidechain"): "relativeSideChain",
        ("relative", "mainchain"): "relativeMainChain",
        ("relative", "polar"): "relativePolar",
        ("relative", "apolar"): "relativeApolar",
    }
    attr = mapping[(sasa_kind, sasa_field)]
    return float(getattr(residue_area, attr))


def compute_structure_sasa_for_requested_residues(
    pdb_path: Path,
    requested_residues: Iterable[ResidueKey],
) -> Tuple[Dict[ResidueKey, dict], List[ResidueKey]]:
    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    residues = result.residueAreas()

    requested = set(requested_residues)
    out: Dict[ResidueKey, dict] = {}
    missing: List[ResidueKey] = []

    for chain_id, res_no, res_name_upper in requested:
        chain_map = residues.get(chain_id)
        if not chain_map:
            missing.append((chain_id, res_no, res_name_upper))
            continue

        residue_area = chain_map.get(str(res_no))
        if residue_area is None:
            residue_area = chain_map.get(int(res_no)) if res_no.isdigit() else None

        if residue_area is None:
            missing.append((chain_id, res_no, res_name_upper))
            continue

        residue_type = normalize_residue_3letter(getattr(residue_area, "residueType", ""), output_case="upper")
        if residue_type and residue_type != res_name_upper:
            missing.append((chain_id, res_no, res_name_upper))
            continue

        out[(chain_id, res_no, res_name_upper)] = {
            "sasa_abs_total": float(getattr(residue_area, "total")),
            "sasa_rel_total": float(getattr(residue_area, "relativeTotal")),
            "residue_area": residue_area,
        }

    return out, missing


def parse_topk_labels(s: str) -> List[str]:
    if not s:
        return []
    return [normalize_residue_3letter(x, output_case="title") for x in s.split("|") if x.strip()]


def assign_bins(
    values: Sequence[float], mode: str, num_bins: int, custom_bins: Optional[str]
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot assign bins with no values")

    if mode == "custom":
        if not custom_bins:
            raise ValueError("--custom-bins is required when --binning custom")
        edges = np.array([float(x.strip()) for x in custom_bins.split(",") if x.strip()], dtype=float)
        if len(edges) < 2:
            raise ValueError("Custom bins must provide at least two edges")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("Custom bin edges must be strictly increasing")
        return edges

    if num_bins < 1:
        raise ValueError("--num-bins must be >= 1")

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if math.isclose(vmin, vmax):
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
        return np.array([vmin - eps, vmax + eps], dtype=float)

    if mode == "equal_width":
        return np.linspace(vmin, vmax, num_bins + 1)

    if mode == "equal_frequency":
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(arr, quantiles)
        edges = np.unique(edges)
        if edges.size < 2:
            eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
            return np.array([vmin - eps, vmax + eps], dtype=float)
        return edges

    raise ValueError(f"Unknown binning mode '{mode}'")


def compute_binned_accuracies(rows: Sequence[dict], bin_edges: np.ndarray) -> List[dict]:
    supports = np.zeros(len(bin_edges) - 1, dtype=int)
    top1_hits = np.zeros(len(bin_edges) - 1, dtype=int)
    top3_hits = np.zeros(len(bin_edges) - 1, dtype=int)
    top5_hits = np.zeros(len(bin_edges) - 1, dtype=int)

    for row in rows:
        val = row.get("sasa_selected")
        actual = normalize_residue_3letter(row.get("actual") or "", output_case="title")
        predicted = normalize_residue_3letter(row.get("predicted") or "", output_case="title")
        if val in (None, "") or not actual:
            continue

        sasa_value = float(val)
        idx = np.searchsorted(bin_edges, sasa_value, side="right") - 1
        if idx == len(bin_edges) - 1:
            idx = len(bin_edges) - 2
        if idx < 0 or idx >= len(supports):
            continue

        topk = parse_topk_labels(row.get("topk_labels") or "")

        supports[idx] += 1
        if predicted == actual:
            top1_hits[idx] += 1
        if actual in topk[:3]:
            top3_hits[idx] += 1
        if actual in topk[:5]:
            top5_hits[idx] += 1

    summary = []
    for i in range(len(supports)):
        support = int(supports[i])
        summary.append(
            {
                "bin_index": i,
                "bin_left": float(bin_edges[i]),
                "bin_right": float(bin_edges[i + 1]),
                "support": support,
                "top1_accuracy": (top1_hits[i] / support) if support else None,
                "top3_accuracy": (top3_hits[i] / support) if support else None,
                "top5_accuracy": (top5_hits[i] / support) if support else None,
            }
        )
    return summary


def save_accuracy_plot(
    binned: Sequence[dict],
    bin_edges: np.ndarray,
    plot_path: Path,
    sasa_kind: str,
    sasa_field: str,
) -> None:
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    labels = [f"{x:.3g}" for x in centers]
    x = np.arange(len(centers))

    y1 = [np.nan if b["top1_accuracy"] is None else b["top1_accuracy"] for b in binned]
    y3 = [np.nan if b["top3_accuracy"] is None else b["top3_accuracy"] for b in binned]
    y5 = [np.nan if b["top5_accuracy"] is None else b["top5_accuracy"] for b in binned]
    support = [b["support"] for b in binned]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, marker="o", label="top-1")
    plt.plot(x, y3, marker="o", label="top-3")
    plt.plot(x, y5, marker="o", label="top-5")

    for i, s in enumerate(support):
        plt.annotate(str(s), (x[i], 0.02), fontsize=8, ha="center", va="bottom", alpha=0.7)

    title = f"Accuracy vs {sasa_kind.capitalize()} {sasa_field.capitalize()} Solvent Accessibility"
    plt.title(title)
    plt.xlabel("Solvent accessibility bins")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()


def format_bin_labels(bin_edges: np.ndarray) -> List[str]:
    labels = []
    for i in range(len(bin_edges) - 1):
        labels.append(f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f})")
    return labels


def build_grouped_correctness_by_bin(
    rows: Sequence[dict], bin_edges: np.ndarray
) -> Tuple[Dict[str, Dict[str, List[int]]], List[str], List[int]]:
    bin_labels = format_bin_labels(bin_edges)
    grouped = {
        label: {
            "top1": [],
            "top3": [],
            "top5": [],
        }
        for label in bin_labels
    }
    supports = [0] * len(bin_labels)

    for row in rows:
        val = row.get("sasa_selected")
        actual = normalize_residue_3letter(row.get("actual") or "", output_case="title")
        predicted = normalize_residue_3letter(row.get("predicted") or "", output_case="title")
        if val in (None, "") or not actual:
            continue

        sasa_value = float(val)
        idx = np.searchsorted(bin_edges, sasa_value, side="right") - 1
        if idx == len(bin_edges) - 1:
            idx = len(bin_edges) - 2
        if idx < 0 or idx >= len(bin_labels):
            continue

        topk = parse_topk_labels(row.get("topk_labels") or "")
        top1_correct = int(predicted == actual)
        top3_correct = int(actual in topk[:3])
        top5_correct = int(actual in topk[:5])

        label = bin_labels[idx]
        grouped[label]["top1"].append(top1_correct)
        grouped[label]["top3"].append(top3_correct)
        grouped[label]["top5"].append(top5_correct)
        supports[idx] += 1

    return grouped, bin_labels, supports


def save_grouped_correctness_boxplot(
    bin_to_metric_values: Dict[str, Dict[str, List[int]]],
    bin_labels: Sequence[str],
    supports: Sequence[int],
    out_path: Path,
) -> None:
    metrics = ["top1", "top3", "top5"]
    colors = {"top1": "#F44336", "top3": "#4CAF50", "top5": "#2196F3"}
    offsets = {"top1": -0.25, "top3": 0.0, "top5": 0.25}

    fig, ax = plt.subplots(figsize=(14, 6))
    centers = np.arange(1, len(bin_labels) + 1, dtype=float)
    legend_handles = []

    for metric in metrics:
        metric_data = []
        metric_positions = []
        for center, bin_label in zip(centers, bin_labels):
            values = bin_to_metric_values[bin_label][metric]
            if not values:
                continue
            metric_data.append(values)
            metric_positions.append(center + offsets[metric])

        if not metric_data:
            continue

        box = ax.boxplot(
            metric_data,
            positions=metric_positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[metric])
            patch.set_alpha(0.65)
        for median in box["medians"]:
            median.set_color("black")

        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[metric], alpha=0.65, label=metric))

    ax.set_title("Prediction Correctness by Solvent Accessibility Bin")
    ax.set_xlabel("Solvent Accessibility Bin")
    ax.set_ylabel("Correctness (0/1)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(centers)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)

    if legend_handles:
        ax.legend(handles=legend_handles, title="Metric")

    for center, support in zip(centers, supports):
        ax.text(center, -0.08, f"n={support}", ha="center", va="top", fontsize=8, alpha=0.75)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_augmented_predictions_csv(rows: Sequence[dict], output_csv: Path, original_fields: Sequence[str]) -> None:
    appended = [
        "chain_id",
        "res_no",
        "res_name",
        "sasa_abs_total",
        "sasa_rel_total",
        "sasa_selected",
        "sasa_kind",
        "sasa_field",
    ]
    fieldnames = list(original_fields) + [f for f in appended if f not in original_fields]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc="Writing output CSV", leave=False):
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def main() -> None:
    args = parse_args()

    pred_rows = load_predictions_csv(args.predictions_csv)
    if not pred_rows:
        raise ValueError(f"No prediction rows found in {args.predictions_csv}")

    original_fields = list(pred_rows[0].keys())
    manifest_rows = load_manifest_csv(args.test_manifest)
    manifest_lookup, fallback_lookup = build_manifest_lookup(manifest_rows)

    cache = load_cache(args.cache_json) if args.cache_json else {}
    requested_by_structure: Dict[str, set] = defaultdict(set)

    matched = 0
    unmatched = 0
    fallback_warnings = 0

    for row in tqdm(pred_rows, desc="Matching predictions"):
        try:
            (
                manifest_row,
                structure_id,
                chain_id,
                res_no,
                res_name_upper,
                warning,
            ) = resolve_prediction_row(row, manifest_lookup, fallback_lookup, args.strict_match)
        except Exception as exc:
            if args.fail_fast:
                raise
            unmatched += 1
            print(f"[WARN] Could not parse/match row: site={row.get('site')} error={exc}")
            continue

        if warning:
            fallback_warnings += 1
            print(
                f"[WARN] {warning}; structure={structure_id} chain={chain_id} res_no={res_no} res_name={res_name_upper}"
            )

        if manifest_row is None:
            unmatched += 1
            msg = (
                f"[WARN] unmatched prediction: structure={structure_id} chain={chain_id} "
                f"res_no={res_no} res_name={res_name_upper}"
            )
            if args.fail_fast:
                raise ValueError(msg)
            print(msg)
            row["chain_id"] = chain_id
            row["res_no"] = res_no
            row["res_name"] = res_name_upper
            row["sasa_abs_total"] = ""
            row["sasa_rel_total"] = ""
            row["sasa_selected"] = ""
            row["sasa_kind"] = args.sasa_kind
            row["sasa_field"] = args.sasa_field
            continue

        matched += 1
        manifest_chain = (manifest_row.get("chain_id") or "").strip()
        manifest_res_no = str(manifest_row.get("res_no") or "").strip()
        manifest_res_name = normalize_residue_3letter(manifest_row.get("res_name") or "", output_case="upper")

        row["chain_id"] = manifest_chain
        row["res_no"] = manifest_res_no
        row["res_name"] = manifest_res_name
        row["sasa_kind"] = args.sasa_kind
        row["sasa_field"] = args.sasa_field

        requested_by_structure[structure_id].add((manifest_chain, manifest_res_no, manifest_res_name))

    missing_pdb_files: List[str] = []
    missing_residues_reported = 0
    residue_scores: Dict[Tuple[str, str, str, str], dict] = {}

    for structure_id, requested_residues in tqdm(requested_by_structure.items(), desc="Processing structures"):
        pdb_path = pdb_path_for_structure(structure_id, args.pdb_root)
        if not pdb_path.exists():
            missing_pdb_files.append(structure_id)
            msg = f"[WARN] Missing PDB file for structure {structure_id}: {pdb_path}"
            if args.fail_fast:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        cache_struct = cache.setdefault(structure_id, {})
        needed = []
        for chain_id, res_no, res_name in requested_residues:
            residue_key = f"{chain_id}|{res_no}|{res_name}"
            if residue_key not in cache_struct:
                needed.append((chain_id, res_no, res_name))

        if needed:
            extracted, missing = compute_structure_sasa_for_requested_residues(pdb_path, needed)
            for chain_id, res_no, res_name in missing:
                missing_residues_reported += 1
                msg = (
                    f"[WARN] residue not found in SASA output: structure={structure_id} "
                    f"chain={chain_id} res_no={res_no} res_name={res_name}"
                )
                if args.fail_fast:
                    raise ValueError(msg)
                print(msg)

            for (chain_id, res_no, res_name), data in extracted.items():
                residue_area = data.pop("residue_area")
                data["sasa_selected"] = select_sasa_value(residue_area, args.sasa_kind, args.sasa_field)
                cache_struct[f"{chain_id}|{res_no}|{res_name}"] = data

        for chain_id, res_no, res_name in requested_residues:
            row_key = (structure_id, chain_id, res_no, res_name)
            cache_key = f"{chain_id}|{res_no}|{res_name}"
            if cache_key in cache_struct:
                residue_scores[row_key] = cache_struct[cache_key]

    if args.cache_json:
        save_cache(args.cache_json, cache)

    rows_with_sasa = 0
    for row in tqdm(pred_rows, desc="Annotating predictions"):
        structure_id = (row.get("structure_name") or "").strip()
        chain_id = (row.get("chain_id") or "").strip()
        res_no = str(row.get("res_no") or "").strip()
        res_name = normalize_residue_3letter(row.get("res_name") or "", output_case="upper")

        score = residue_scores.get((structure_id, chain_id, res_no, res_name))
        if not score:
            row["sasa_abs_total"] = ""
            row["sasa_rel_total"] = ""
            row["sasa_selected"] = ""
            continue

        row["sasa_abs_total"] = score["sasa_abs_total"]
        row["sasa_rel_total"] = score["sasa_rel_total"]
        row["sasa_selected"] = score["sasa_selected"]
        rows_with_sasa += 1

    write_augmented_predictions_csv(pred_rows, args.output_csv, original_fields)

    rows_for_bins = [r for r in pred_rows if r.get("sasa_selected") not in (None, "") and (r.get("actual") or "").strip()]
    if not rows_for_bins:
        raise ValueError("No rows available for plotting (need valid actual + sasa_selected)")

    sasa_values = [float(r["sasa_selected"]) for r in rows_for_bins]
    bin_edges = assign_bins(sasa_values, args.binning, args.num_bins, args.custom_bins)
    binned = compute_binned_accuracies(rows_for_bins, bin_edges)

    if args.plot_mode == "line":
        save_accuracy_plot(binned, bin_edges, args.plot_png, args.sasa_kind, args.sasa_field)
    else:
        grouped, bin_labels, supports = build_grouped_correctness_by_bin(rows_for_bins, bin_edges)
        save_grouped_correctness_boxplot(grouped, bin_labels, supports, args.plot_png)

    if args.summary_json:
        summary = {
            "total_prediction_rows": len(pred_rows),
            "matched_residues": matched,
            "unmatched_residues": unmatched,
            "residues_scored": rows_with_sasa,
            "missing_pdb_files": sorted(set(missing_pdb_files)),
            "missing_residues_reported": missing_residues_reported,
            "fallback_warnings": fallback_warnings,
            "selected_sasa_metric": {
                "sasa_kind": args.sasa_kind,
                "sasa_field": args.sasa_field,
            },
            "plot_mode": args.plot_mode,
            "binning": {
                "mode": args.binning,
                "num_bins": args.num_bins,
                "custom_bins": args.custom_bins,
                "bin_edges": [float(x) for x in bin_edges.tolist()],
            },
            "per_bin": binned,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if args.verbose:
        print(f"Wrote augmented predictions CSV: {args.output_csv}")
        print(f"Wrote SASA accuracy plot: {args.plot_png}")
        if args.summary_json:
            print(f"Wrote summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
