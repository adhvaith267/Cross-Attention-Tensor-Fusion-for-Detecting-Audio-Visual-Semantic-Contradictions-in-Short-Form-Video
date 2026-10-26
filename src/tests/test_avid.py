"""AVID-Bench subset selection on a tiny fake annotation set (no network)."""

import json

import pytest

from av_contradiction.data.avid import ANNOTATIONS, load_pairs, select_subset, source_of


def _write(root, split, videos, segs=2, types=("SEMANTIC_DIVERGENCE", "LIP_SYNC")):
    folder = "train_7239_segments" if split == "train" else "test_1171_segments"
    path = root / ANNOTATIONS[split]
    path.parent.mkdir(parents=True, exist_ok=True)
    remote = {}
    with open(path, "w", encoding="utf-8") as fh:
        for v in videos:
            for k in range(segs):
                for kind, exists in (("negative", "Yes"), ("positive", "No")):
                    vp = f"{folder}/{kind}/{v}_{k}.mp4"
                    gt = {"exists": exists, "injection_type": types[k % len(types)] if exists == "Yes" else "", "inconsistency_point": "x"}
                    fh.write(json.dumps({"qa_id": f"{kind[:3]}_{v}_{k}", "video_path": vp, "ground_truth": gt}) + "\n")
                    remote[vp] = {"size": 1_000_000, "sha256": "0" * 64}
    return remote


def test_source_of_keeps_underscores_and_dashes():
    assert source_of("train_7239_segments/negative/--x4_dom-r14c_12.mp4") == "--x4_dom-r14c"


def test_selection_rules(tmp_path):
    remote = _write(tmp_path, "train", [f"tr{i}" for i in range(30)]) | _write(tmp_path, "test", [f"te{i}" for i in range(10)])
    rows, rep = select_subset(tmp_path, remote, {"train": 10, "val": 5, "test": 5}, budget_gb=1.0, categories=("SEMANTIC_DIVERGENCE",))
    assert rep["selected_pairs"] == {"train": 10, "val": 5, "test": 5}
    assert rep["excluded"]["official_train"]["category_LIP_SYNC"] == 30
    for r in rows:  # label follows the official 'exists' field: negative folder = injected = contradictory
        assert r["label"] == ("contradictory" if "/negative/" in r["video_path"] else "consistent")
        assert (r["split"] == "test") == (r["official_split"] == "test")
    splits_of = {}
    for r in rows:
        splits_of.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits_of.values())  # a source video never spans splits
    for split in ("train", "val", "test"):
        labels = [r["label"] for r in rows if r["split"] == split]
        assert labels.count("consistent") == labels.count("contradictory")  # balanced by pairing, no duplication


def test_budget_drops_pairs(tmp_path):
    remote = _write(tmp_path, "train", [f"tr{i}" for i in range(30)]) | _write(tmp_path, "test", [f"te{i}" for i in range(10)])
    rows, rep = select_subset(tmp_path, remote, {"train": 10, "val": 5, "test": 5}, budget_gb=0.02)
    assert rep["selected_bytes"] <= 0.02e9
    assert sum(rep["selected_pairs"].values()) == 10


def test_inconsistent_folder_and_label_rejected(tmp_path):
    p = tmp_path / ANNOTATIONS["train"]
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"qa_id": "a", "video_path": "train_7239_segments/positive/v_0.mp4",
                             "ground_truth": {"exists": "Yes"}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pairs(p, "train")
