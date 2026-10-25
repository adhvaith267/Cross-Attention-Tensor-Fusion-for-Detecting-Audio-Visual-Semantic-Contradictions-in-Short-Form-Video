"""Audio-visual semantic contradiction detection with cross-attention tensor fusion."""

__version__ = "0.1.0"

LABELS: tuple[str, ...] = ("consistent", "contradictory")
LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
