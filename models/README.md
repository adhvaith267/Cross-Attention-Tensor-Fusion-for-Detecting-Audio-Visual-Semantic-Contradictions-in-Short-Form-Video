# Released models

Put a trained model here to ship it with the repository, one folder per model:

```
models/<name>/
  best.pt              # weights + training config (no optimizer state)
  calibration.json     # temperature fitted on validation
  metrics_test.json    # held-out test results shown in the dashboard
  best_model.json      # which preset/seed was selected and why
```

This is exactly the content of `best_model.zip` from the Kaggle notebook: unzip it into
`models/<name>/`. The dashboard lists every `models/*/best.pt` automatically.

Released here:

| folder | model | trained on | test macro-F1 |
|---|---|---|---|
| `avid_swap/` | `concat` (default) | audio-swap benchmark: natural audio, each clip with its own and with another clip's audio | 0.665 |
| `avid/` | `xattn_tensor_global` | AVID-Bench subset: original vs injected (edited) audio | 0.950 |

`default.txt` names the folder the dashboard selects first.
