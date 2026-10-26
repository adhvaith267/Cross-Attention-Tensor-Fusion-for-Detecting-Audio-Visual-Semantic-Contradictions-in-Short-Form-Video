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
