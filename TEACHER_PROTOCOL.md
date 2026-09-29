# Teacher Route Protocol

## Fixed route

The teacher route is evaluated in three stages:

```text
A: Larger temporal CNN + relative spectral branch
B: CNN + 2-layer Transformer + relative spectral branch
C: Three-seed ensemble of the selected teacher
```

The student remains the frozen Gen5 baseline: temporal CNN plus relative band power.

## Data protocol

- Development data is the original train and val subjects combined.
- Use fixed 8-fold `GroupKFold`; windows from one subject never cross folds.
- Stage A uses one seed for architecture screening.
- Stage B uses one seed only if A does not meet the teacher gate.
- Stage C uses seeds 42, 43, and 44 after the teacher architecture is fixed.
- Test is not read during A, B, or C selection.

## Teacher gate

The teacher must beat the Gen5 development reference (`BA 0.673 +/- 0.046`, `AUC 0.729 +/- 0.076`) with stable fold performance before distillation.

Report for every fold:

```text
balanced accuracy, macro-F1, ROC-AUC, validation subjects, confusion matrix
```

The final teacher report also includes per-subject metrics and mean/std across seeds. Only a fixed teacher proceeds to distillation.

## Stage A model

```text
window-local z-score EEG
  -> three temporal CNN branches (kernel 7, 15, 31)
  -> channel fusion

relative delta/theta/alpha/beta/gamma power
  -> spectral MLP

temporal + spectral features -> classifier
```

Stage A uses class-balanced cross entropy, AdamW, gradient clipping, and validation BA early stopping.

## Stage B search protocol

Stage B is a development-only hyperparameter search for a multi-scale temporal
CNN, temporal Transformer, and relative spectral-power branch. The test split
is never loaded by the search script.

All methods and trials use the same fixed subject-disjoint `GroupKFold`
partitions. The search and fixed-model verification use eight folds and the
same 15-epoch budget. The
script supports independent random search, Sobol quasi-random search, and
Gaussian-process Bayesian search with expected improvement.

The search space is:

```text
learning_rate: log-uniform 1e-4 to 1e-3
weight_decay:  log-uniform 1e-6 to 1e-3
dropout:       uniform 0.10 to 0.35
d_model:       96 or 128
n_heads:       4 or 8
n_layers:      1 or 2
batch_size:    128 or 256
```

Trials are ranked by `mean BA - 0.25 * BA std`, which favors a strong and
stable configuration. The selected configuration must be re-trained with the
full eight-fold script before it becomes the fixed Teacher B model.

Teacher B acceptance thresholds are:

- mean BA >= 0.72
- BA standard deviation <= 0.05
- mean ROC-AUC >= 0.77
- ROC-AUC standard deviation <= 0.08
- minimum fold BA >= 0.62

## Teacher B v2 frozen configuration

The Bayesian-selected configuration is frozen in
`configs/teacher_b_v2.json` and must not be changed during seed confirmation:

```text
learning_rate = 3.0265400272101283e-4
weight_decay  = 4.084129623824164e-5
dropout       = 0.14111876080331268
d_model       = 128
n_heads       = 4
n_layers      = 1
batch_size    = 128
epochs        = 15
```

The next experiment is a confirmation only: seeds 42, 43, and 44, each with
the fixed eight-fold subject GroupKFold protocol. It reports fold metrics and
per-subject metrics. The frozen teacher is accepted only if:

- mean of the three seed BA values >= 0.72
- standard deviation of the three seed BA values <= 0.05
- no seed mean BA < 0.70
- minimum fold BA across all seeds >= 0.62

## Route B repair: nested threshold calibration

The seed confirmation showed that fixed probability threshold `0.5` still
fails on several subjects. The next diagnostic keeps Teacher B v2 frozen and
uses nested subject GroupKFold. For each outer fold, an inner GroupKFold
produces out-of-fold probabilities from outer-train subjects; the threshold is
selected from those probabilities only, then evaluated once on outer
validation. The test split remains untouched.

The report compares fixed threshold `0.5` with the calibrated threshold. If
calibration improves BA while preserving AUC, the problem is partly decision
calibration. If it does not, the remaining limitation is representation or
cross-subject domain shift and the model structure must be revisited.
