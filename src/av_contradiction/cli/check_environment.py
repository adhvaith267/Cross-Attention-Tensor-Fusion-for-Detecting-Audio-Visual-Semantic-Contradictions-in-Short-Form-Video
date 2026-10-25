"""Report Python/PyTorch/CUDA/FFmpeg/disk status and which optional components are usable.

Usage: python -m av_contradiction.cli.check_environment [--check-network] [--json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import urllib.request

from av_contradiction.utils.device import environment_report
from av_contradiction.utils.reproducibility import package_versions

HF_MODELS = ("openai/clip-vit-base-patch32", "laion/clap-htsat-unfused", "openai/whisper-small")


def ffmpeg_status() -> dict:
    from av_contradiction.data.preprocessing import find_ffmpeg

    try:
        exe = find_ffmpeg()
        out = subprocess.run([exe, "-version"], capture_output=True, timeout=20).stdout.decode(errors="replace")
        return {"path": exe, "version": out.splitlines()[0] if out else "unknown"}
    except Exception as exc:
        return {"path": None, "error": str(exc)}


def hf_cache_status() -> dict:
    status = {}
    try:
        from huggingface_hub import try_to_load_from_cache

        for m in HF_MODELS:
            status[m] = "cached" if isinstance(try_to_load_from_cache(m, "config.json"), str) else "not cached"
    except ImportError:
        status = {m: "huggingface_hub not installed" for m in HF_MODELS}
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-network", action="store_true", help="Probe https://huggingface.co reachability")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = environment_report()
    report["packages"] = package_versions(("torch", "numpy", "scikit-learn", "transformers", "imageio-ffmpeg", "streamlit", "tensorboard"))
    report["optional"] = {m: importlib.util.find_spec(m) is not None for m in ("transformers", "streamlit", "tensorboard")}
    report["ffmpeg"] = ffmpeg_status()
    usage = shutil.disk_usage(".")
    report["disk_free_gb"] = round(usage.free / 1024**3, 1)
    report["hf_cache"] = hf_cache_status()
    if args.check_network:
        try:
            urllib.request.urlopen("https://huggingface.co", timeout=10)
            report["network"] = "huggingface.co reachable"
        except Exception as exc:
            report["network"] = f"unreachable: {exc}"
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"Python {report['python']} | torch {report['torch']} | CUDA available: {report['cuda_available']}")
    for g in report["gpus"]:
        print(f"  GPU {g['index']}: {g['name']} ({g['memory_gb']} GB)")
    print(f"Recommended hardware profile: {report['recommended_profile']}")
    print(f"FFmpeg: {report['ffmpeg'].get('version') or report['ffmpeg'].get('error')}")
    print(f"Free disk (cwd): {report['disk_free_gb']} GB")
    print("Optional packages:", ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in report["optional"].items()))
    print("Pretrained weights in local HF cache:")
    for k, v in report["hf_cache"].items():
        print(f"  {k}: {v}")
    if "network" in report:
        print("Network:", report["network"])
    if not report["cuda_available"]:
        print("NOTE: no CUDA GPU detected. CPU works for tests, the synthetic benchmark and inference; "
              "CLIP/CLAP feature extraction on real data will be slow.")


if __name__ == "__main__":
    main()
