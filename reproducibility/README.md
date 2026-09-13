# GUAVA-Net seeded-run records

This directory contains the reproducibility records for the independently seeded GUAVA-Net experiments reported in the manuscript.

## Contents

- `seed_results.csv`: per-seed evaluation metrics for BCData and DeepLIIF. Values are represented on the `[0,1]` scale.
- `split_manifests/`: exact filenames and experimental assignments for both datasets.
- `train_configs/`: saved training-option files for seeds 42, 123, and 2026 on each dataset.
- `test_logs/`: console logs produced by evaluating the prespecified epoch-200 checkpoints.

## Experimental protocol

For each dataset, the model was independently trained from scratch with seeds 42, 123, and 2026. Hyperparameters and the epoch-200 checkpoint were specified before evaluation. The held-out evaluation images were not used for hyperparameter tuning, early stopping, or checkpoint selection.

For BCData, all 385 files from the processed BC-DeepLIIF training release were used for training, and all 66 files from its original validation release were reserved as the held-out evaluation set. For DeepLIIF, all 575 official training files were used for optimization, the 91 official validation files were not used, and all 598 official test files were used only for final evaluation.

The manuscript reports the arithmetic mean and sample standard deviation across the three runs. The run-directory suffixes such as `clean` and `clean2` are local bookkeeping labels and do not denote different model or evaluation settings.

## Summary reported in the manuscript

| Dataset | PixAcc | Dice | IoU | AJI | IHC Quant Diff |
|---|---:|---:|---:|---:|---:|
| BCData | 0.9304 +/- 0.0002 | 0.7686 +/- 0.0007 | 0.6286 +/- 0.0013 | 0.6076 +/- 0.0045 | 0.1410 +/- 0.0070 |
| DeepLIIF | 0.8857 +/- 0.0010 | 0.7601 +/- 0.0013 | 0.6176 +/- 0.0018 | 0.4509 +/- 0.0095 | 0.0871 +/- 0.0022 |
