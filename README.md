# Cross-Attention Tensor Fusion for Detecting Audio-Visual Semantic Contradictions

Give it a short video; it tells you whether what you **hear** is consistent with what you **see**
(a violin sound over a visible guitar, narration describing a red car over a blue one). It returns
`consistent`, `contradictory` or `uncertain`, a calibrated score, and the timestamps it looked at.

> **Status.** The pipeline (data tooling, training, evaluation, dashboard, Kaggle notebook) is
> implemented and tested, but it has **only been trained on a synthetic sanity benchmark**. There
> are no real-data results yet.

## How it works

```
video ─► FFmpeg ─► 16 frames + timestamps ─► CLIP ViT-L/14 ─┐
                └► 2 s audio windows + times ─► CLAP ───────┴► cross-attention ─► tensor fusion ─► label + score
```

1. **Decode** with FFmpeg: 16 frames at exact timestamps and 2 s audio windows (1 s hop).
2. **Encode** frames with frozen CLIP ViT-L/14 and audio with frozen CLAP (`laion/larger_clap_general`).
3. **Fuse**: bidirectional audio↔video cross-attention (timestamp encodings, a temporal-proximity
   bias, a learned "no match" token), then low-rank tensor fusion and a classifier.
4. **Predict**: scores between 0.35 and 0.65 are reported as `uncertain`.

Only the fusion model (~3.2 M parameters) is trained. Encoder features are computed once and cached,
so training itself takes minutes.

## Install

```bash
git clone https://github.com/adhvaith267/Cross-Attention-Tensor-Fusion-for-Detecting-Audio-Visual-Semantic-Contradictions-in-Short-Form-Video.git
cd Cross-Attention-Tensor-Fusion-for-Detecting-Audio-Visual-Semantic-Contradictions-in-Short-Form-Video
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch                 # NVIDIA GPU: use the CUDA command from https://pytorch.org/get-started/locally/
pip install -e ".[encoders,dashboard,dev]"
python -m av_contradiction.cli.download_encoders        # one-off: CLIP ViT-L/14 1.7 GB + CLAP 0.8 GB
python -m av_contradiction.cli.check_environment
```

FFmpeg ships with `imageio-ffmpeg`. Weights are only downloaded by `download_encoders` (or with
`--set encoders.local_files_only=false`); after that everything runs offline.

## Train on Kaggle (free GPU)

`notebooks/kaggle_train.ipynb` does the whole run: clone this repo, download AVHBench, build
leakage-safe splits, extract features, train every model variant over 3 seeds, pick the best on
**validation** macro-F1, report it once on **test**, and zip it for download. Import the notebook,
set *Accelerator → GPU* and *Internet → On*, then *Save Version → Save & Run All*.

It fits comfortably in Kaggle's free tier (~30 GPU-h/week, 12 h sessions, 16 GB GPU):

| step | cost | notes |
|---|---|---|
| feature extraction | ~10 min per 1,000 clips | FFmpeg decoding (CPU) dominates: 1.4 s per 10 s clip per thread, 4 threads in parallel while the GPU encodes in fp16. Resumable: cached clips are skipped. |
| 7 variants × 3 seeds | minutes | fusion models on cached features held in memory |
| disk | ~70 KB per clip | features; well under the 20 GB output limit |

The large encoders cost little on Kaggle because decoding is the bottleneck. Inference runs the same
encoders again, so a local NVIDIA GPU is recommended: they need ~1 GB of VRAM in fp16 (any 4 GB+
card, e.g. an RTX 4050 laptop GPU, works). On a CPU-only laptop they take ~20 s per video (measured); there, set
`encoders.video_model_name=openai/clip-vit-base-patch16` (~4× cheaper) before extracting and training.

## Use a trained model

Released models live in `models/<name>/` and are committed to the repo, so after installing you can
predict straight away without training. To release your own, unzip the notebook's `best_model.zip`
into `models/<name>/` and commit it (see [`models/README.md`](models/README.md)).

```bash
streamlit run src/av_contradiction/serving/dashboard.py                                     # dashboard
python -m av_contradiction.cli.predict --video clip.mp4 --checkpoint models/<name>/best.pt   # JSON
```

The dashboard lists every `best.pt` under `models/` and `output/checkpoints/`, shows its saved test
metrics, and can run a second model (e.g. `audio_only`) side by side.

## Train on your own data

A manifest is a CSV with `sample_id, video_path, label` (`consistent` / `contradictory`) and
optionally `split` and `group_id`. Splits are made per group, so a clip and anything derived from it
never land in different splits. `prepare_dataset` also converts AVHBench and builds VGGSound
within-category audio swaps.

```bash
python -m av_contradiction.cli.prepare_dataset split --manifest my.csv --out data/manifests/manifest.csv
python -m av_contradiction.cli.audit_dataset --manifest data/manifests/manifest.csv --probe
python -m av_contradiction.cli.extract_features --set encoders.local_files_only=false
python -m av_contradiction.cli.run_ablations --seeds 1 2 3      # trains, evaluates, saves best_model/
```

Every setting lives in `src/av_contradiction/config.py` and can be changed with `--set key=value`
or a YAML file passed with `--config`.

**Offline smoke test** (CPU, a few minutes, no downloads): a synthetic colour↔tone benchmark with
parameter-free encoders.

```bash
python -m av_contradiction.cli.make_synthetic_benchmark --scenes 150
python -m av_contradiction.cli.extract_features --config quick.yaml
python -m av_contradiction.cli.run_ablations --config quick.yaml
```

```yaml
# quick.yaml
data: {manifest: data/manifests/synthetic_colortone.csv, root_dir: data/synthetic_colortone}
encoders: {video: handcrafted, audio: handcrafted}
preprocess: {audio_sample_rate: 16000, num_frames: 8, frame_size: 64}
```

## Model variants

| preset | audio | video | cross-attention | fusion | time alignment |
|---|---|---|---|---|---|
| `majority`, `similarity` | non-neural baselines | | | | |
| `audio_only` / `video_only` | ✓ / – | – / ✓ | | concat | ✓ |
| `concat` | ✓ | ✓ | | concat | ✓ |
| `tensor_no_xattn` | ✓ | ✓ | | tensor | ✓ |
| `xattn_concat` | ✓ | ✓ | ✓ | concat | ✓ |
| `xattn_tensor_global` | ✓ | ✓ | ✓ | tensor | |
| **`xattn_tensor` (proposed)** | ✓ | ✓ | ✓ | tensor | ✓ |

## Results

**Real data: none yet.** On the synthetic benchmark (handcrafted encoders, 60 test clips, 3 seeds),
test macro-F1 was `xattn_concat` 0.994 ± 0.010, `xattn_tensor` 0.978 ± 0.010, `concat` 0.728,
`audio_only` 0.525, `video_only` 0.498. Timestamp-aware cross-attention is what mattered there;
tensor fusion gave no measurable gain over concatenation. The training schedule was changed after
first-run test numbers had been seen, so this is not a fully blind result.

## Limitations

* No public dataset has true semantic-contradiction labels. AVHBench's AV-matching labels and
  constructed audio swaps are proxies and may contain shortcuts.
* Frames are encoded one at a time by an image model; motion is seen only through the frame sequence.
* Sarcasm, off-screen sound and background music are not modelled.
* Attention weights and "least visual support" moments are hints, not explanations.
* Calibration is fitted on validation data and does not transfer to very different videos.

## Third-party licenses and responsible use

CLAP is Apache-2.0. OpenAI's CLIP model card limits CLIP to research use. AVHBench states
no license; VGGSound annotations are CC BY 4.0, and the videos remain their uploaders' copyright.
Do not redistribute YouTube-derived videos. This is not a misinformation, deception or deepfake
detector, and its outputs must not be used to label people or content as deceptive.

## Development

```bash
pytest                    # CPU-only unit and end-to-end tests, no downloads
pytest -m integration     # real CLIP + CLAP (weights must be cached)
ruff check src
```

Code is in `src/av_contradiction/` (`data/`, `models/`, `training/`, `evaluation/`, `inference/`,
`cli/`, `serving/`), tests in `src/tests/`.
