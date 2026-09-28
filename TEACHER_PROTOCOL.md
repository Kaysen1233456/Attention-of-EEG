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
