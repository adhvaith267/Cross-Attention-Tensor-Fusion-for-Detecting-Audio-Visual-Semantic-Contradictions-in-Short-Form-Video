"""Download the pretrained CLIP + CLAP weights into the local Hugging Face cache (one-off, ~2.5 GB).

After this, prediction and the dashboard run fully offline.

Usage:
  python -m av_contradiction.cli.download_encoders                              # encoders from the default config
  python -m av_contradiction.cli.download_encoders --checkpoint models/<name>/best.pt   # encoders a model was trained with
  python -m av_contradiction.cli.download_encoders --asr        # also Whisper, for the dashboard's speech transcript
"""

from __future__ import annotations

import argparse

from av_contradiction.config import Config
from av_contradiction.evaluation.evaluate import config_from_checkpoint
from av_contradiction.training.checkpointing import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="Use the encoder names stored in this checkpoint")
    ap.add_argument("--asr", action="store_true", help="Also download the Whisper speech-to-text model (~290 MB)")
    args = ap.parse_args()
    cfg = config_from_checkpoint(load_checkpoint(args.checkpoint)) if args.checkpoint else Config()
    enc = cfg.encoders
    from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel, CLIPVisionModelWithProjection

    # The tokenizers are only needed for the dashboard's text view (zero-shot tags).
    if enc.video == "clip":
        print(f"Downloading {enc.video_model_name} ...")
        CLIPVisionModelWithProjection.from_pretrained(enc.video_model_name)
        AutoTokenizer.from_pretrained(enc.video_model_name)
    if enc.audio == "clap":
        print(f"Downloading {enc.audio_model_name} ...")
        ClapModel.from_pretrained(enc.audio_model_name)
        ClapFeatureExtractor.from_pretrained(enc.audio_model_name)
        AutoTokenizer.from_pretrained(enc.audio_model_name)
    if args.asr:
        from transformers import pipeline

        print(f"Downloading {cfg.inference.asr_model_name} ...")
        pipeline("automatic-speech-recognition", model=cfg.inference.asr_model_name)
    print("Done. The encoders are cached; inference now works offline.")


if __name__ == "__main__":
    main()
