"""Miniature end-to-end pipeline: render -> split -> extract -> train -> resume -> evaluate -> predict."""

import json
from pathlib import Path

import pytest

from av_contradiction.config import replace_section
from av_contradiction.data.manifest import resolve_video_path
from av_contradiction.evaluation.ablations import run_suite
from av_contradiction.evaluation.evaluate import evaluate_checkpoint
from av_contradiction.inference.predictor import Predictor
from av_contradiction.training.checkpointing import load_checkpoint
from av_contradiction.training.trainer import NonFiniteLossError, train_from_config


def test_checkpoint_contents(trained_checkpoint):
    ck = load_checkpoint(trained_checkpoint)
    for key in ("optimizer_state", "scheduler_state", "epoch", "global_step", "seed", "manifest_hash", "feature_hash",
                "versions", "best_metric", "label_names", "config"):
        assert key in ck
    assert ck["config"]["preprocess"]["num_frames"] == 4


def test_resume_continues_training(mini_dataset):
    cfg = replace_section(mini_dataset["cfg"], training={"epochs": 1})
    cfg.experiment_name = "resume_test"
    first = train_from_config(cfg)
    assert first.epochs_run == 1
    cfg2 = replace_section(cfg, training={"epochs": 2})
    second = train_from_config(cfg2, resume=first.last_checkpoint)
    assert second.epochs_run == 2 and second.global_step > first.global_step
    assert len(second.history) == 2


def test_evaluate_writes_reports(trained_checkpoint, mini_dataset):
    res = evaluate_checkpoint(trained_checkpoint, device="cpu")
    out = Path(mini_dataset["cfg"].evaluation.output_dir) / mini_dataset["cfg"].experiment_name
    assert (out / "metrics_test.json").exists() and (out / "confusion_test.png").exists()
    assert json.loads((out / "metrics_test.json").read_text())["split"] == "test"
    assert 0.0 <= res["metrics"]["macro_f1"] <= 1.0
    assert (trained_checkpoint.parent / "calibration.json").exists()


def test_predictor_end_to_end(trained_checkpoint, mini_dataset):
    pred = Predictor(trained_checkpoint, "cpu")
    path = Path(mini_dataset["cfg"].data.root_dir) / mini_dataset["rows"][1]["video_path"]
    r = pred.predict(path)
    assert r.label in ("consistent", "contradictory", "uncertain")
    assert r.timing_ms["decode_ms"] > 0
    assert len(r.evidence.frames) == mini_dataset["cfg"].preprocess.num_frames
    assert r.evidence.attention_links  # the proposed model exposes audio->video attention


def test_nan_guard(mini_dataset, monkeypatch):
    cfg = replace_section(mini_dataset["cfg"], training={"epochs": 1, "max_nan_steps": 0})
    cfg.experiment_name = "nan_test"
    from av_contradiction.models import classifier

    orig = classifier.AVContradictionModel.forward

    def bad_forward(self, batch, return_attention=False):
        out = orig(self, batch, return_attention)
        out["logits"] = out["logits"] * float("nan")
        return out

    monkeypatch.setattr(classifier.AVContradictionModel, "forward", bad_forward)
    with pytest.raises(NonFiniteLossError):
        train_from_config(cfg)


def test_ablation_suite_runs(mini_dataset):
    cfg = replace_section(mini_dataset["cfg"], training={"epochs": 1})
    cfg.experiment_name = "suite_test"
    res = run_suite(cfg, ["majority", "similarity", "audio_only", "xattn_tensor"], seeds=[0])
    statuses = {r["preset"]: r["status"] for r in res["runs"]}
    assert statuses == {"majority": "ok", "similarity": "ok", "audio_only": "ok", "xattn_tensor": "ok"}
    assert Path(res["output_dir"], "ablation_summary.md").exists()
    best = res["best"]
    assert best["preset"] in ("audio_only", "xattn_tensor")  # only neural runs are eligible
    assert best["val_macro_f1"] == max(r["val_macro_f1"] for r in res["runs"] if "val_macro_f1" in r)
    released = Path(res["output_dir"], "best_model", "best.pt")
    assert Path(res["output_dir"], "best_model", "metrics_test.json").exists()
    assert "optimizer_state" not in load_checkpoint(released)  # slim file for shipping in models/
    video = resolve_video_path(mini_dataset["rows"][0], mini_dataset["cfg"].data.root_dir)
    assert Predictor(released, "cpu").predict(video).label in ("consistent", "contradictory", "uncertain")
