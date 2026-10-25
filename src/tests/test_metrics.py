import numpy as np
import pytest

from av_contradiction.training.losses import compute_class_weights
from av_contradiction.training.metrics import (
    compute_metrics,
    expected_calibration_error,
    fit_temperature,
    softmax,
)

NAMES = ["consistent", "contradictory"]


def test_metrics_known_values():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    m = compute_metrics(labels, probs, NAMES)
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["per_class"]["contradictory"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["consistent"]["recall"] == 0.5
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["roc_auc"] == 1.0
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.36, 0.04, 0.09]))


def test_auc_undefined_single_class():
    m = compute_metrics(np.array([1, 1]), np.array([[0.2, 0.8], [0.1, 0.9]]), NAMES)
    assert m["roc_auc"] is None and m["pr_auc"] is None


def test_ece_zero_when_calibrated():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 1])) == 0.0
    assert expected_calibration_error(np.array([[0.9, 0.1]]), np.array([1])) == pytest.approx(0.9)


def test_temperature_scaling_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 2000)
    clean = np.where(labels[:, None] == np.arange(2), 1.0, -1.0)
    noisy_labels = np.where(rng.random(2000) < 0.2, 1 - labels, labels)  # 20% label noise
    t = fit_temperature(clean * 10, noisy_labels)
    assert t > 3  # overconfident logits must be softened
    p = softmax(clean * 10, t)
    assert expected_calibration_error(p, noisy_labels) < expected_calibration_error(softmax(clean * 10), noisy_labels)


def test_class_weights_auto():
    assert compute_class_weights(np.array([0, 1] * 10), 2, "auto") is None
    w = compute_class_weights(np.array([0] * 30 + [1] * 10), 2, "auto")
    assert w is not None and w[1] > w[0] and w.mean().item() == pytest.approx(1.0)
    assert compute_class_weights(np.array([0] * 30 + [1] * 10), 2, "none") is None
