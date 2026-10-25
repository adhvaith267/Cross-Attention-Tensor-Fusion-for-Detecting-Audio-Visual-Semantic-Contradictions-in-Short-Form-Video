"""Non-neural baselines and baseline model presets.

Neural baselines (audio-only, video-only, concat, cross-attention without tensor fusion) are the
same ``AVContradictionModel`` with ablation switches; see ``evaluation/ablations.py:PRESETS``.
"""

from __future__ import annotations

import numpy as np


class MajorityClassBaseline:
    """Always predicts the most frequent training label (probability = training class frequency)."""

    def fit(self, labels: np.ndarray, num_classes: int = 2) -> MajorityClassBaseline:
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        self.prior = counts / counts.sum()
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.tile(self.prior, (n, 1))


class ProjectedSimilarityBaseline:
    """Similarity baseline for encoders that do NOT share an embedding space (e.g. CLIP vs CLAP).

    Scoring rule:
      1. Fit ridge regression  W: mean audio embedding -> mean video embedding  on *consistent*
         training clips only.
      2. score(x) = 1 - cos(W a_x, v_x)   (higher = less consistent).
      3. Threshold tau chosen on the validation split to maximise macro-F1.
      4. Reported "probability" = sigmoid((score - tau) / s) with s = std of validation scores.
         This is an uncalibrated monotone score, not a probability estimate.
    """

    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def fit(self, audio: np.ndarray, video: np.ndarray, labels: np.ndarray, consistent_id: int = 0) -> ProjectedSimilarityBaseline:
        a, v = self._norm(audio[labels == consistent_id]), self._norm(video[labels == consistent_id])
        self.a_mean = a.mean(0)
        a_c = a - self.a_mean
        self.W = np.linalg.solve(a_c.T @ a_c + self.ridge * np.eye(a.shape[1]), a_c.T @ v)
        return self

    def score(self, audio: np.ndarray, video: np.ndarray) -> np.ndarray:
        pred = self._norm((self._norm(audio) - self.a_mean) @ self.W)
        return 1.0 - (pred * self._norm(video)).sum(1)

    def tune_threshold(self, audio: np.ndarray, video: np.ndarray, labels: np.ndarray) -> ProjectedSimilarityBaseline:
        from sklearn.metrics import f1_score

        s = self.score(audio, video)
        candidates = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
        f1s = [f1_score(labels, (s > t).astype(int), average="macro", zero_division=0) for t in candidates]
        self.tau = float(candidates[int(np.argmax(f1s))])
        self.scale = float(s.std() + 1e-6)
        return self

    def predict_proba(self, audio: np.ndarray, video: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(self.score(audio, video) - self.tau) / self.scale))
        return np.stack([1 - p, p], axis=1)
