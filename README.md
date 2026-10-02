# SNN-TORCH

A stellar-classification research example using two single-step LIF layers
and a linear output head, with a conventional MLP baseline on the same split.
The LIF state resets on each forward call. This is not a temporal spike-sequence
experiment, energy benchmark or production astronomy classifier.

## Runnable Synthetic Demo

In a dedicated Python 3.12 environment, for CPU-only testing:

```sh
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python demo.py
python -m pip check
```

The demo trains both paths on generated numeric data without downloading a
dataset and reports coverage/epoch counts, not stellar-classification quality.
Tests also execute the notebook on a temporary synthetic CSV using a headless
plotting backend. GPU/MPS tests are explicitly skipped when unavailable; CPU
CI is not proof of accelerator compatibility.

## Corrected Protocol

- Stratified row split before learned preprocessing. Label mapping/scaling fit
  only training rows; optional seeded SMOTE operates only on those rows.
- Both models use the same split, processed training data, shuffle seed, hidden
  widths and fixed epoch/learning-rate budget. They are not separately tuned.
- No test-score monitoring each epoch. Training-loss history is recorded and
  the held-out data is evaluated once per model after training.
- Evaluation and prediction include the final incomplete batch, even when the
  whole test set is smaller than a training batch. Empty/partial evaluation is
  rejected, not reported as a successful score.
- Evaluation/prediction accept standard batched `DataLoader` instances with a
  sequential sampler, default collation and ordered delivery. Use
  `DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)` or the
  test loader returned by `make_loaders`. Replacement/shuffled/custom samplers,
  custom collation and unordered delivery are rejected: matching the number of
  rows alone cannot prove coverage, and predictions must retain dataset order.
  This is a supported-loader contract, not protection from a malicious dataset.
- Input tensors and labels move to the configured model device. Move the model
  before constructing its optimizer; a mismatch gives an explicit error.
- Cross-entropy is averaged over samples rather than giving each batch equal
  weight. Confusion matrices retain all model classes. Undefined AUC is `None`,
  not zero. Per-class undefined sensitivity/specificity is also `None`; macro
  values average only defined denominators. Metrics are fractions, not percent.
- Unlabelled inference uses `CustomDataset(..., is_train=False)` and `predict`;
  it does not try to index missing labels. Supply exactly the trained feature
  schema and fitted scaler. Leave out the target column.

Core behavior lives in `star_classifier.py`; the notebook calls it instead of
maintaining a second training implementation. Old saved outputs were cleared
because they came from the earlier, defective evaluation path.

## Real Data

Place a reviewed local `star_classification.csv` at
`data/star_classification.csv`, then open `SNN_Star_classification.ipynb` from
the repository root. No CSV, trained weights or data downloader is included.
The notebook drops the original seven identifier columns and expects a `class`
target plus finite numeric features. Class counts must support a stratified
split; SMOTE needs at least two training examples in each class.

The repository does not establish the dataset's source/version or redistribution
permission. Verify those before use. A random row split does not prevent
duplicate-object or survey-group leakage, and some columns may not be available
at deployment time. Audit object/group independence and feature availability
before interpreting any real score. No corrected real-data metrics are claimed.

Record input provenance, split, seeds, preprocessing, environment and full
per-class results for reproducibility. Dependency ranges are not a lock file;
same-seed CPU tests do not promise identical results across hardware/releases.
The original snapshot had no root LICENSE; this repair does not grant a new
license or settle ownership of third-party code/data.
