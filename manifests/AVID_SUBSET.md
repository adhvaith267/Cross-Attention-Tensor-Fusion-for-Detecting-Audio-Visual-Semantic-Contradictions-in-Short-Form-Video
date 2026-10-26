# AVID-Bench segment subset (4.87 GB)

A reproducible, leakage-safe subset of **AVID-Bench** for training and evaluating this project's
audio-visual contradiction model on a Kaggle GPU. The exact file list is
[`avid_subset.csv`](avid_subset.csv); the media are downloaded from the official host, never stored in git.

## Source and access

| | |
|---|---|
| Dataset | AVID-Bench, [github.com/czx1220/AVID-bench](https://github.com/czx1220/AVID-bench) |
| Media host | ModelScope dataset [`czx13927247324/AVID`](https://modelscope.cn/datasets/czx13927247324/AVID), the host documented in the official README |
| Hugging Face | [`Chen-15/AVID`](https://huggingface.co/datasets/Chen-15/AVID) hosts the dataset card and 4 test-side annotation files only, **no media** |
| Access method | official `modelscope` SDK: `HubApi.get_dataset_files` (listing, sizes, SHA-256) and `dataset_file_download` (one file at a time) |
| License | Apache-2.0 per the dataset card and repository. The clips derive from YouTube videos whose copyright stays with their uploaders: research use only, do not redistribute the media. |

## What was downloaded

* `annotations/*.jsonl`: all 8 official annotation files (63 MB). The selection uses
  `train_segments.jsonl` and `test_segments.jsonl`.
* 2,600 individual segment MP4s from `train_7239_segments/{positive,negative}/` and
  `test_1171_segments/{positive,negative}/`. No full videos or archives were downloaded.

## Labels (verified against the annotations)

Folder names are **not** our labels; the label comes from `ground_truth.exists`:

| official file | `exists` | meaning | our label |
|---|---|---|---|
| `*_segments/positive/<vid>_<k>.mp4` | `No` | original audio | `consistent` |
| `*_segments/negative/<vid>_<k>.mp4` | `Yes` | audio **injected** with an inconsistency of `injection_type` | `contradictory` |

A positive and a negative file with the same name show the same clip: their video streams are
byte-identical (checked by MD5) and only the audio differs. AVID-Bench has 8 injection types:

| type | used | why |
|---|---|---|
| `SEMANTIC_DIVERGENCE`, `EMOTION_MISMATCH`, `BACKGROUND_CONFLICT`, `BACKGROUND_SOUND`, `VOICE_IDENTITY` | ✓ | the audio's content contradicts what is shown |
| `VOLUME_FLUCTUATION` | – | a loudness glitch, not a semantic contradiction |
| `TEMPORAL_SHIFT`, `LIP_SYNC` | – | timing offsets; not semantic, and below this model's resolution (2 s audio windows, 16 frames per 5–30 s clip) |

All negatives are **synthetic injections** made by the dataset authors, so this measures detection of
injected inconsistencies, a proxy for naturally occurring semantic contradictions.

## Selection

* **Matched pairs only**: each selected segment is used twice, once with its original audio
  (consistent) and once with injected audio (contradictory). The visuals alone cannot predict the
  label, and classes are balanced without duplicating anything.
* **One pair per source video**, so 1,300 different source videos, with an equal number of pairs per
  injection type (round-robin, seed 0).
* **Splits**: `test` comes only from the official test split; `train` and `val` come only from the
  official train split (AVID has no validation split), on disjoint source videos. Official train and
  test share no source video.
* Only annotated records are used. The host also stores 24,305 unannotated `positive/` training
  files; they have no official label and were not used.

| split | pairs | segments | consistent | contradictory | source videos | per injection type |
|---|---|---|---|---|---|---|
| train | 1,000 | 2,000 | 1,000 | 1,000 | 1,000 | 200 |
| val | 150 | 300 | 150 | 150 | 150 | 30 |
| test | 150 | 300 | 150 | 150 | 150 | 30 |

Test video classes (official `videoclass`): Voiceover 60, Scenic 60, Active Speaker 30 pairs.

Excluded candidates (`data/avid/selection_report.json`):

| reason | official train | official test |
|---|---|---|
| injected segment has no annotated consistent counterpart | 24,305 | 0 |
| `VOLUME_FLUCTUATION` | 1,435 | 514 |
| `TEMPORAL_SHIFT` | 744 | 569 |
| `LIP_SYNC` | 551 | 589 |
| not selected (budget / one pair per video) | the rest | the rest |

## Verification

`python -m av_contradiction.cli.prepare_avid verify` (report: `data/avid/verification_report.json`):

| check | result |
|---|---|
| files present with the official size and SHA-256 | 2,600 / 2,600 |
| decodable video stream + middle frame + full audio track | 2,600 / 2,600 (0 corrupt, 0 missing audio or video) |
| label equals the official `ground_truth.exists` of the same `qa_id` and path | 2,600 / 2,600 |
| source videos in more than one split / duplicate paths or ids | 0 / 0 |
| test rows from official train, or train/val rows from official test | 0 |
| clip duration | 5.0–30.0 s, median 13.4 s |
| near-silent audio (RMS < 1e-3) | 10 files, all `consistent` (original audio silent; the injection added sound) |
| disk usage | **4.87 GB** total: 4.79 GB media, 61 MB annotations, 12 MB cached file listing |

The 10 near-silent clips are a tiny shortcut (silence implies consistent); they are kept because they
are the official data, and the predictor returns `uncertain` for fully silent audio anyway.

## Reproduce

```bash
pip install -e ".[avid]"
python -m av_contradiction.cli.prepare_avid select     # annotations + remote listing + selection (rewrites the same manifest)
python -m av_contradiction.cli.prepare_avid download   # resumable; skips files whose SHA-256 already matches
python -m av_contradiction.cli.prepare_avid verify     # decode check, label re-check, leakage check
```

`download` fetches only the files listed in `manifests/avid_subset.csv`, verifies size and SHA-256
against the official listing, retries transient errors (4 tries, exponential backoff), deletes
partial files, and refuses to start a file that would push `data/avid/` past `--budget-gb`
(default 10). Each file is logged in `data/avid/download_log.jsonl`. No credentials are needed.

## Use on Kaggle

The manifest uses paths relative to the dataset folder, so the folder can move anywhere.

**Option A (recommended): upload once as a private Kaggle Dataset.** Zip `data/avid/` without
compression (the MP4s are already compressed):

```bash
python -c "import zipfile,pathlib as p; r=p.Path('data/avid'); z=zipfile.ZipFile('avid_subset.zip','w',zipfile.ZIP_STORED,allowZip64=True); [z.write(f,f.relative_to(r).as_posix()) for f in r.rglob('*') if f.is_file() and 'remote_listing' not in f.parts]; z.close()"
```

On kaggle.com: *Datasets → New Dataset*, upload `avid_subset.zip`, name it e.g. `avid-subset`
(the name must contain `avid`), keep it **private**. In `notebooks/kaggle_train.ipynb` attach it with
*Add Input*; the notebook finds it under `/kaggle/input/` and uses it as `data.root_dir`.

**Option B: download on Kaggle.** With *Internet → On*: `pip install -e ".[avid]"` and
`python -m av_contradiction.cli.prepare_avid --root /kaggle/working/avid download`, then
use `data.root_dir=/kaggle/working/avid`. On the machine that built this subset it ran at ~0.5 files/s
(~75 min); speed from Kaggle is untested.

## Limitations

* Contradictions are synthetic audio injections; injection artefacts (cuts, level changes) could be
  learned as shortcuts. Compare against the `audio_only` baseline: a strong audio-only score signals one.
* Segment-level annotations carry no timestamps (segments are pre-cut); the full-video annotations'
  time ranges were not used.
* The injection descriptions (`notes`) are written by the dataset authors and were not re-checked by hand.
