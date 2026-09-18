"""Train-only memorization diagnostic. Never loads validation/test or checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train import AttentionConfig, build_model, apply_ablation_variant, set_seed, project_root
from attention_model.data.dataset import EEGWindowDataset


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default=str(project_root / 'configs/teacher_ear_saad.yaml'))
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--per-class', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.steps < 1 or a.per_class < 1 or a.lr <= 0:
        p.error('steps, per-class and lr must be positive')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    set_seed(a.seed)
    torch.set_float32_matmul_precision('highest')
    cfg = AttentionConfig.from_yaml(a.config)
    apply_ablation_variant(cfg, 'B1')
    cfg.model.pretrained_backbone_path = None
    cfg.model.use_pretrained_backbone = False
    cfg.model.freeze_pretrained_backbone = False
    data = Path(cfg.data.data_dir)
    if not data.is_absolute():
        data = project_root / data
    raw = np.load(data / 'train_waveforms.npy', allow_pickle=False)
    labels = np.load(data / 'train_labels.npy', allow_pickle=False)
    if not np.isfinite(raw).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Expected finite waveforms and binary 0/1 labels')
    rng = np.random.default_rng(a.seed)
    ids = np.concatenate([rng.choice(np.flatnonzero(labels == c), a.per_class, replace=False) for c in (0, 1)])
    ds = EEGWindowDataset(raw, labels, normalize=cfg.data.normalize,
                          normalization_mode=cfg.data.normalization_mode,
                          clip_std=cfg.data.normalize_clip_std)
    x = torch.stack([ds[int(i)]['waveform'] for i in ids]).to(a.device)
    y = torch.tensor(labels[ids], dtype=torch.long, device=a.device)
    model = build_model(cfg).to(a.device)
    # The shared builder currently clamps teacher dropout to >=0.1.
    # Explicitly disable every Dropout in this diagnostic only.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    history = []
    passed = False
    for step in range(1, a.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x)['logits'], y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 10 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                logits = model(x)['logits']
                ce = torch.nn.functional.cross_entropy(logits, y).item()
                prob = logits.softmax(-1)[:, 1]
                pred = logits.argmax(-1)
                acc = (pred == y).float().mean().item()
                recalls = [(pred[y == c] == c).float().mean().item() for c in (0, 1)]
            row = dict(step=step, loss=ce, accuracy=acc, recalls=recalls,
                       probability_std=prob.std().item(), gradient_norm=float(norm), lr=a.lr)
            if not np.isfinite([ce, row['probability_std'], row['gradient_norm']]).all():
                raise FloatingPointError('Non-finite diagnostic')
            history.append(row)
            print(json.dumps(row), flush=True)
            passed = acc >= .98 and ce < .1
            payload = dict(status='passed' if passed else 'in_progress', test_used=False,
                           scope='training subset memorization; NOT generalization',
                           indices=ids.tolist(), args=vars(a), config=cfg.to_dict(),
                           dropout_override=0.0, amp=False, weight_decay=0.0,
                           torch_version=torch.__version__, history=history)
            (out / 'diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
            if passed:
                break
    payload['status'] = 'passed' if passed else 'budget_exhausted'
    (out / 'diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
