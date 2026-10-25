"""Plots and tables generated strictly from saved experiment outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion(cm: list[list[int]], labels: list[str], path: str | Path, title: str = "") -> None:
    plt = _plt()
    cm_a = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm_a, cmap="Blues")
    for i in range(cm_a.shape[0]):
        for j in range(cm_a.shape[1]):
            ax.text(j, i, str(cm_a[i, j]), ha="center", va="center", color="black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(probs: np.ndarray, labels: np.ndarray, path: str | Path, n_bins: int = 10, title: str = "") -> None:
    plt = _plt()
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (conf >= lo) & (conf <= hi)
        if sel.any():
            centers.append(conf[sel].mean())
            accs.append(correct[sel].mean())
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(centers, accs, "o-")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_predictions(path: str | Path, ids: list[str], labels: np.ndarray, probs: np.ndarray, label_names: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "label", "pred"] + [f"p_{n}" for n in label_names])
        for i, sid in enumerate(ids):
            w.writerow([sid, label_names[labels[i]], label_names[int(probs[i].argmax())]] + [f"{p:.6f}" for p in probs[i]])


SUMMARY_KEYS = ("macro_f1", "accuracy", "roc_auc", "pr_auc", "ece", "brier")


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group run results by preset and compute mean/std over seeds for each summary metric."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by[r["preset"]].append(r)
    rows = []
    for preset, runs in by.items():
        row: dict[str, Any] = {"preset": preset, "n_seeds": len(runs), "status": ",".join(sorted({r.get("status", "ok") for r in runs}))}
        ok = [r for r in runs if r.get("status", "ok") == "ok"]
        for k in SUMMARY_KEYS:
            vals = [r["metrics"][k] for r in ok if r.get("metrics", {}).get(k) is not None]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        row["params"] = ok[0].get("params") if ok else None
        row["latency_ms"] = float(np.mean([r["latency_ms"] for r in ok if r.get("latency_ms") is not None])) if ok and any(r.get("latency_ms") is not None for r in ok) else None
        rows.append(row)
    return rows


def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.3f} ± {std:.3f}" if std is not None else f"{mean:.3f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    head = "| preset | seeds | macro-F1 | accuracy | ROC-AUC | PR-AUC | ECE | params | latency (ms/clip, fusion only) | status |"
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        lat = f"{r['latency_ms']:.2f}" if r.get("latency_ms") is not None else "n/a"
        lines.append(
            f"| {r['preset']} | {r['n_seeds']} | {_fmt(r['macro_f1_mean'], r['macro_f1_std'])} | "
            f"{_fmt(r['accuracy_mean'], r['accuracy_std'])} | {_fmt(r['roc_auc_mean'], r['roc_auc_std'])} | "
            f"{_fmt(r['pr_auc_mean'], r['pr_auc_std'])} | {_fmt(r['ece_mean'], r['ece_std'])} | "
            f"{r['params'] if r['params'] is not None else 'n/a'} | {lat} | {r['status']} |"
        )
    return "\n".join(lines)


def plot_comparison(rows: list[dict[str, Any]], path: str | Path, metric: str = "macro_f1") -> None:
    plt = _plt()
    rows = [r for r in rows if r.get(f"{metric}_mean") is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(rows)), 3.5))
    means = [r[f"{metric}_mean"] for r in rows]
    errs = [r[f"{metric}_std"] or 0 for r in rows]
    ax.bar(range(len(rows)), means, yerr=errs, capsize=3)
    ax.set_xticks(range(len(rows)), [r["preset"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
