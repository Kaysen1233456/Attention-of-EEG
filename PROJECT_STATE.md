# EEG Attention Project State

Updated: 2026-09-18

## Goal

Build an Ear-SAAD attention-decoding system that uses the NeurIPT-inspired
components during research, but deploys a lightweight dual-branch CNN on an
ear-worn device. The target is cross-subject generalization, not a score from
a random window split.

## Agreed Mainline

The project uses a supervised teacher-student route:

```text
Stage 1: supervised teacher architecture validation
  EEG -> bilateral encoder -> IILP pooling -> classifier

Stage 2: supervised teacher training
Transformer initialized from scratch -> left/right IILP pooling ->
  [L, R, |L-R|, L*R] -> SwiGLU classifier

Stage 3: knowledge distillation for deployment
  teacher logits and bilateral features -> lightweight dual-branch CNN student
```

The deployed model is the CNN student, not the Transformer teacher. The AAMP
checkpoint from the previous attempt is rejected and must not be loaded. The
3D embedding, PMoE, and IILP remain architectural candidates, but each must
earn its place through supervised validation.

Do not load Transformer encoder weights into CNN convolution layers. Their
parameters have different shapes and meanings. The only valid direct transfer
is between identical `MiniNeurIPT` encoders.

## Dataset and Evaluation Contract

- Dataset: `data/processed/ear_saad`.
- Shape: train `(6426, 19, 100)`, validation `(2142, 19, 100)`, test `(2142, 19, 100)`.
- Task: balanced binary attention decoding.
- Split: train/validation/test subjects are disjoint.
- Input: 9 left-ear channels and 10 right-ear channels; keep the listed order
  and the 19 channel coordinates unchanged.
- Normalization: training-subject statistics only, clipped to 8 standard deviations.
- Model selection/search: training and validation subjects only.
- Test set: evaluate once after the strategy and hyperparameters are fixed.
- Report window-level balanced accuracy, macro-F1, and ROC-AUC. Do not choose
  checkpoints from `subject_level_best_threshold`; that metric optimizes the
  threshold on the validation labels and is diagnostic only.

## Trusted Results

### Previous CNN baseline

- `artifacts/ear_saad_a10_bs128`: best validation balanced accuracy `0.5341`.
- Test balanced accuracy `0.5037`.
- A previous 30-trial CNN Sobol search reached validation balanced accuracy
  `0.5612`; it is validation-only and not a final test result.

### AAMP pretraining (rejected)

Checkpoint:

`artifacts/pretrain_aamp_ear_saad_d96/best_pretrain_model.pt`

Pretraining contract:

- 19 channels, 100 samples/window, `d_model=96`, 8 attention heads, 4 layers,
  `d_ff=384`, temporal pool `4`.
- Best reported validation pretraining loss: `0.4845`.
- Independent validation reconstruction check: masked L1 `0.4756`; zero-output
  baseline `0.5568`; copying the masked input baseline `0.5205`.
- Although reconstruction beats the simple baselines, the learned features
  are judged unusable for the downstream task and are excluded from all new
  experiments. This is not a downstream classification result.

The pretraining loss includes a small PMoE load-balancing term, so it is not
identical to pure reconstruction L1. AAMP masking is stochastic, so exact
losses vary by mask realization.

## Changes Made Today

### Teacher architecture

- Added `MiniNeurIPTClassifier` as the supervised teacher path.
- Added `LearnableIILPPooling` in
  `src/attention_model/models/mini_neuript.py`.
- The teacher encodes all 19 channels, then pools left and right channel-time
  tokens with separate learned attention weights. It fuses `L`, `R`, `|L-R|`,
  and `L*R` before a SwiGLU classifier.
- Existing AAMP encoder checkpoint keys remain unchanged and load strictly.
- Teacher forward was verified with the existing checkpoint. Parameter count:
  `1,991,725`.

### AAMP and configuration contract

- `MiniNeurIPT` now accepts the full AAMP configuration: mask ratios,
  80/10/10 replacement ratios, percentile range, amplitude mode, and temporal
  pooling factor.
- `scripts/pretrain_aamp.py` forwards all those YAML values.
- Added `configs/teacher_ear_saad.yaml`, which fixes the teacher architecture
  to the pretrained encoder contract (`d_model=96`, 4 layers, 8 heads,
  `d_ff=384`) and supplies the full Ear-SAAD data configuration.

### Fine-tuning strategy

- Added `encoder_freeze_epochs` and `encoder_lr_scale` to `TrainingConfig`.
- The trainer recognizes models with an `encoder` and uses two parameter groups:
  classifier/IILP head at base learning rate and encoder at
  `base_lr * encoder_lr_scale`.
- During the initial freeze stage, encoder parameters do not receive gradients
  and the encoder runs in evaluation mode. This avoids immediate catastrophic
  forgetting of AAMP features.

### Search strategy

- Added `log_float` support to Sobol and Bayesian search. Learning rate and
  weight decay must be sampled in log space.
- Teacher search uses discrete batch sizes `[32, 64, 96, 128]`.
- Fixed `scripts/search_hyperparams.py` so its `--data` default no longer
  overwrites a YAML `data_dir`.

## Interrupted Searches: Do Not Treat as Results

The B5 search in `artifacts/search_ablation_B5_sobol` was interrupted on
2026-09-18 after five trial artifacts were written. It started before B0-B4,
so it is protocol-external exploratory output and must not be used for any
architecture or hyperparameter decision. Its directory contains an explicit
marker for this status.

Two searches were intentionally stopped because they exposed a training-policy
problem rather than a useful hyperparameter optimum:

- `artifacts/search_teacher_ear_saad_sobol8`: full encoder fine-tuning from
  epoch 1. The first two trials peaked at validation balanced accuracy `0.4935`
  and `0.5014`, then overfit strongly.
- `artifacts/search_teacher_ear_saad_sobol6_progressive`: used five frozen
  epochs then unfroze the encoder. Trial 0 reached `0.5593` at frozen epoch 1,
  then degraded during later frozen epochs and after unfreezing.

The important conclusion is that the current AAMP encoder is fragile under
supervised adaptation and must first be evaluated as a fixed feature extractor.
The `0.5593` value is one validation observation, not a final claim or a
test-set result.

## Next Work, In Order

1. Train a supervised teacher from random initialization. The current teacher
   configuration disables the rejected AAMP checkpoint and trains the encoder,
   IILP, and SwiGLU head end to end.
2. Run controlled architecture ablations on validation subjects: CNN baseline,
   bilateral Transformer without PMoE, then PMoE/IILP additions. Change one
   component at a time and keep the test split untouched.
3. Run at least three seeds for the best teacher configuration and report
   validation mean and standard deviation.
4. After a teacher materially exceeds the CNN baseline, train the student with
   `CE(labels) + KD(teacher logits, temperature) + bilateral feature loss`.
   The reusable `BilateralFeatureProjector` and `DistillationLoss` are now in
   `src/attention_model/training/distillation.py`; projection heads remain
   training-only and are excluded from deployment.
6. If the teacher cannot consistently beat the CNN validation baseline, pause
   architectural expansion and audit label/window alignment, subject-specific
   label orientation, missing channels, and distribution shift.

## Execution Ledger (2026-09-18)

The controlled single-seed B0-B5 ablation was completed on 2026-09-18 with
validation-only evaluation. No test evaluation was performed.

| Stage | Scope | State |
|---|---|---|
| 1 | Supervised teacher from scratch | B0-B5 controlled validation completed |
| 2 | Controlled architecture ablations | Completed for B0-B5, seed 42 |
| 3 | Select the best validated teacher architecture | B1 is the current candidate; multi-seed pending |
| 4 | Three-seed teacher reproducibility | Protocol defined; not started |
| 5 | Teacher-to-CNN distillation | Loss/projector interface implemented; training not started |

The next Codex should begin with Stage 1 using
`configs/teacher_ear_saad.yaml`, keep the test split untouched, and record all
search results under a new artifact directory. The rejected checkpoint must
not appear in any new configuration or artifact selection.

## Commands

Teacher configuration:

```bash
python scripts/train.py \
  --config configs/teacher_ear_saad.yaml \
  --device cuda
```

Teacher Sobol search:

```bash
python scripts/search_hyperparams.py \
  --method quasi_random \
  --config configs/teacher_ear_saad.yaml \
  --n-trials 8 \
  --search-epochs 25 \
  --num-workers 4 \
  --output artifacts/search_teacher_from_scratch \
  --seed 42
```

The current configuration sets `training.encoder_freeze_epochs: 0` and leaves
the encoder randomly initialized. Do not point the final teacher run at the
test set for selection.

## Verification Performed

- `python -m compileall -q src scripts` passes.
- Existing AAMP checkpoint strictly loaded into the teacher encoder during the
  previous implementation, but it is excluded from the current route.
- Teacher output shapes and left/right IILP attention normalization were checked.
- Progressive encoder freeze/unfreeze parameter groups were smoke-tested.
- B0 construction reproduces the trusted CNN parameter count: `227,536`.
- Train/validation/test datasets expose trial IDs; all `(subject, trial)` groups
  in the current Ear-SAAD data have a single consistent label.
- Training and search no longer evaluate test unless `--evaluate-test` is set.

## Execution Ledger Update (2026-09-18)

- Added `configs/development_folds.json` as the versioned fixed 3-fold
  development manifest over the 12 train/val subjects.
- Corrected the multi-seed CLI so every seed runs in a fresh subprocess with
  fresh model, loaders, optimizer, scheduler, and AMP scaler.
- Training now records run identity and evaluates train and validation at the
  same selected checkpoint; this is the source for the train/validation gap.
- Non-finite ROC-AUC and evaluation metrics now fail loudly instead of being
  converted to zero.
- The README now describes the supervised random-init teacher route and the
  rejected AAMP checkpoint policy.
- Verification after these changes: 57 tests passed and `compileall` passed.
- No new teacher search or test-set evaluation was launched in this change.

## Supervised Ablation Protocol

The architecture ablation is B0-B5. Every run uses the same subject-disjoint
train/validation split, normalization, seed `42`, training budget, and
validation selection metric. The test split remains untouched.

| Variant | Increment | Minimum validation requirement |
|---|---|---|
| B0 | Existing dual-branch CNN | Reproduce `0.5341` balanced accuracy |
| B1 | Random-init Transformer, no PMoE, mean pooling, plain MLP | At least `0.5341` |
| B2 | B1 + independent learned left/right IILP pooling | At least B1 + `0.010` |
| B3 | B2 + `|L-R|` and `L*R` bilateral fusion | At least B2 + `0.010` |
| B4 | B3 + PMoE | At least B3; no regression for adoption |
| B5 | B4 + SwiGLU classifier | At least B4 + `0.010` |

These are decision gates, not achieved results. Keep a component only when it
clears its gate and does not show an obvious train/validation gap. If a later
component fails, retain the strongest earlier variant.

Run each variant separately:

```bash
python scripts/train.py --config configs/teacher_ear_saad.yaml --ablation B0 --epochs 25 --seeds 42 --device cuda --output artifacts/ablation/B0
python scripts/train.py --config configs/teacher_ear_saad.yaml --ablation B1 --epochs 25 --seeds 42 --device cuda --output artifacts/ablation/B1
python scripts/train.py --config configs/teacher_ear_saad.yaml --ablation B2 --epochs 25 --seeds 42 --device cuda --output artifacts/ablation/B2
python scripts/train.py --config configs/teacher_ear_saad.yaml --ablation B3 --epochs 25 --seeds 42 --device cuda --output artifacts/ablation/B3
python scripts/train.py --config configs/teacher_ear_saad.yaml --ablation B4 --epochs 25 --seeds 42 --device cuda --output artifacts/ablation/B4
python scripts/train.py --config configs/teacher_ear_saad.yaml --ablation B5 --epochs 25 --seeds 42 --device cuda --output artifacts/ablation/B5
```

After the runs, plot validation balanced accuracy with:

```bash
python scripts/plot_ablation_results.py \
  --manifest configs/ablation_manifest.example.json \
  --output artifacts/ablation_comparison.png
```

The chart compares achieved validation balanced accuracy against each variant's
minimum target. A run is not considered usable unless it also has validation
macro-F1 at least `0.50`, ROC-AUC at least `0.55`, and a train/validation
balanced-accuracy gap no larger than `0.10`. These are engineering gates for
this dataset, not scientific claims. Report each run's balanced accuracy,
macro-F1, ROC-AUC, best epoch, parameter count, and train/validation gap. Do
not put test metrics in this chart.

## Hyperparameter Search After Ablation

Select the strongest B variant using validation only. Then search its learning
rate, weight decay, dropout, batch size, and only afterward encoder size. Do
not search hyperparameters for every B variant; that would make the component
comparison unfair and unnecessarily expensive.

Sobol search:

```bash
python scripts/search_hyperparams.py \
  --method quasi_random \
  --config configs/teacher_ear_saad.yaml \
  --ablation B5 \
  --n-trials 16 \
  --search-epochs 25 \
  --num-workers 4 \
  --output artifacts/search_ablation_B5_sobol \
  --seed 42
```

Replace `B5` with the selected variant. Bayesian refinement uses the best
Sobol configuration and remains validation-only:

```bash
python scripts/search_hyperparams.py \
  --method bayesian \
  --config configs/teacher_ear_saad.yaml \
  --ablation B5 \
  --n-trials 12 \
  --n-initial 5 \
  --search-epochs 25 \
  --output artifacts/search_ablation_B5_bayesian \
  --seed 42
```

After fixing the architecture and hyperparameters, run seeds `42`, `43`, and
`44` and report mean/std on validation subjects. Only then evaluate the test
split once. No test result may influence B selection or hyperparameter search.
