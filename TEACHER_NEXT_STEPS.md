# Teacher recovery and improvement plan

## Objective

Keep the supervised, randomly initialized teacher -> bilateral features/logits
-> lightweight CNN student route. Do not reuse the rejected AAMP checkpoint.
The teacher target is subject-wise development BA >= 0.60, supported by
multiple folds and seeds, followed by an independently evaluated frozen model.
No current result guarantees that target.

## Evidence and corrections

- B1 single-split ablation BA: 0.5780. Search trial 12 BA: 0.6373.
- Previously reported multi-seed validation BA: 0.6304 +/- 0.0227.
  This is provisional: train_multi_seed replaces the optimizer without
  rebuilding its scheduler and resets modules instead of constructing a fresh
  model. It is not a faithful replication of the search training procedure.
- Existing checkpoint test BA: 0.5317 +/- 0.0211; AUC: 0.5117.
  These describe those checkpoints, not proof that the teacher approach fails.
- Low subject-specific AUC does not prove reversed labels or distribution
  shift. These are hypotheses requiring source-data verification.
- The processed data are 20 Hz, 19 channels, 5-second windows. Metadata reports
  missing-channel zero filling and interpolated NaNs. Labels identify attended
  speaker 1/2; their relation to spatial side must be verified from source.
- The encoder averages groups of four embedded time steps before attention.
  Information loss is a testable hypothesis, not an established root cause.

## Ordered execution

Progress checkpoint (2026-09-18): original trial 12 is reproduced exactly
through the search objective (histories and checkpoint tensors), and all
development waveforms/labels are verified against released MAT trials.
The seed-42 three-fold B1 baseline is complete: BA 0.5357 / 0.5326 / 0.5550,
mean 0.5411. See PROJECT_STATE.md and artifacts/B1_corrected_folds_v1.
Do not expand this weak baseline to seeds 43/44 before controlled design
comparisons. Low same-checkpoint training BA means that "overfitting" alone
is not an adequate diagnosis. First compare normalization and missingness
handling, then temporal_pool=2/1, changing one factor at a time. Keep the
same folds, seed, 25-epoch budget and numerical precision for comparisons.
The ordinary training CLI's `high` precision differs from the exactly
reproduced search objective's `highest`; align or explicitly verify that
before mixing results from those two entry points.

Normalization controls completed (2026-09-18): window-local normalization
collapsed to mean BA 0.5004 and is rejected. Zeroing all-zero channel windows
after train-subject normalization produced mean BA 0.5411, matching the
reference; this proxy control neither improves B1 nor supports a missingness
shortcut explanation. Artifacts: `artifacts/B1_normalization_controls_v1`.
Proceed with `temporal_pool=2` using train-subject normalization and
`zero_channel_policy=retain`. Run `temporal_pool=1` only if pool 2 provides
a stable development improvement that warrants its substantially higher cost.

1. Repair experiment validity before new searches.
   Use fresh model, loaders, optimizer, scheduler and AMP scaler per seed.
   Save effective per-seed configuration, seed, learning-rate history and
   checkpoint identity. Export the winning trial's effective 25-epoch config.
   Confirm seed 42 reproduces the search procedure before seeds 43/44.
   Evaluate training and validation in eval mode at the same checkpoint when
   reporting a generalization gap. Online training accuracy is not that gap.
   Make development loaders exclude test entirely. Strict checkpoint loading
   and model construction must remain shared between train and evaluation.

2. Audit development data and provenance.
   Trace conversion from original trials to arrays, channel order, speaker
   identity/side mapping, sample timing, video condition and missingness.
   Record per-subject/trial amplitude statistics, missing-channel masks,
   interpolation fractions and duplicate windows. Never flip labels based on
   predictive scores. At 20 Hz, do not propose recovering alpha/beta bands
   absent from the released signal. Preserve the original processed arrays.

3. Establish subject-wise development folds over the 12 train+val subjects.
   Fix three folds of eight training and four validation subjects before runs.
   Keep every subject and trial in one partition; fit normalization on each
   fold's training subjects. Use seed 42 to screen configurations and seeds
   42/43/44 for finalists. Report fold-wise and subject-wise BA, class recalls,
   F1, AUC, parameter count and same-checkpoint train/validation metrics.
   Seed variation alone does not estimate uncertainty across new subjects.

4. Improve one factor at a time against a corrected B1 reference.
   First compare train-fitted normalization with window-local normalization
   and missing-channel-aware handling. Window-local normalization can remove
   useful amplitude differences, so retain it only on development evidence.
   Then compare temporal_pool=4, 2, 1 at fixed width/depth. If warranted, test
   a learned temporal convolution front end before attention. Next compare
   a two-layer encoder with the four-layer reference to reduce overfitting.
   Use widths divisible by both three (spatial embedding) and head count;
   do not silently change attention heads. Add modest amplitude/channel
   augmentation only after confirming label preservation and missing masks.
   Whole-target-subject normalization is a separate adaptation protocol and
   must specify what unlabeled data are available at deployment.

5. Search only the strongest validated teacher design.
   Use a bounded 16-trial search over learning rate, decay, dropout and batch
   size, scored across fixed development folds. Include the reference config.
   Do not choose one lucky fold or seed. Investigate threshold calibration
   only with development labels; it cannot restore poor ranking ability.
   Balanced window labels do not justify class weighting automatically.
   PMoE, IILP and SwiGLU remain optional candidates requiring measured gains.

6. Apply a teacher-to-student gate.
   Require development mean BA >= 0.60 across folds and seeds; report spread
   and weak subjects rather than hiding them in the average. Also require
   useful class recalls and F1/AUC, and comparison with the CNN reference
   under the same folds. Only then start distillation and compare the student
   with supervised-only CNN training under the same protocol.

## Test reporting

The existing test subjects have been evaluated, including historical CNN
results. Keep them out of development decisions and label any later reuse
as reuse of an exposed holdout. New independent subjects are preferred for
an unbiased final claim; if unavailable, state that limitation and report
development cross-validation honestly. Do not reshuffle exposed subjects and
call them a fresh blind test.

## Implementation ownership

- scripts/train.py, training/trainer.py: fresh per-seed runs and diagnostics.
- scripts/search_hyperparams.py: effective config export and fold objectives.
- data/dataset.py: development-only loading, fold stats and missing metadata.
- models/mini_neuript.py: isolated temporal representation experiments.
- evaluation/metrics.py and scripts/evaluate.py: consistent frozen inference.
- PROJECT_STATE.md: append experimental evidence and decisions after each gate.

First deliverable: corrected experiment runner plus a development data audit
and fixed fold manifest. Do not launch another large search before these pass.
