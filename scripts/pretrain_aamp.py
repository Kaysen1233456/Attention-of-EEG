"""
AAMP 自监督预训练入口脚本（阶段二）

用法：
    python scripts/pretrain_aamp.py --data data/processed --output artifacts/pretrain
    python scripts/pretrain_aamp.py --config configs/pretrain.yaml

预训练完成后，可以用 train.py 进行微调，或用蒸馏脚本把知识蒸馏到小模型。
"""
import sys
import argparse
import hashlib
import json
import random
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np

from attention_model.config import AttentionConfig
from attention_model.models import MiniNeurIPT
from attention_model.data import EEGDataset, SyntheticEEGDataset
from attention_model.training.trainer import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="AAMP 自监督预训练")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--temporal-pool", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unlabeled-splits",
        choices=("train_only", "train_val", "all"),
        default="train_val",
        help="train_only=搜索阶段；train_val=正式开发预训练；all=包含test的transductive协议",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("fixed_epochs", "validation"),
        default="fixed_epochs",
        help="train_val/all 默认固定epoch；validation仅用于独立train/val实验",
    )
    return parser.parse_args()


class UnlabeledWaveformDataset(Dataset):
    """Drop labels explicitly so the pretraining loop cannot consume them."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return {"waveform": item["waveform"], "index": index}


def _stable_mask_evaluation(model, loader, device, use_amp, seed):
    """Evaluate with a fixed mask stream without changing training RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return evaluate_pretrain(model, loader, device, use_amp)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def pretrain_one_epoch(model, loader, optimizer, device, scaler, use_amp):
    """预训练一个 epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        waveforms = batch["waveform"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(waveforms, apply_mask=True)
            loss = output["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate_pretrain(model, loader, device, use_amp):
    """评估预训练（重建损失）"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        waveforms = batch["waveform"].to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(waveforms, apply_mask=True)
            loss = output["loss"]
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.config:
        config = AttentionConfig.from_yaml(args.config)
        config.data.data_dir = args.data or config.data.data_dir
        config.output.output_dir = args.output or config.output.output_dir
        config.training.epochs = args.epochs if args.epochs is not None else config.training.epochs
        config.training.batch_size = args.batch_size if args.batch_size is not None else config.training.batch_size
        config.training.learning_rate = args.lr if args.lr is not None else config.training.learning_rate
        config.training.weight_decay = args.weight_decay if args.weight_decay is not None else config.training.weight_decay
        config.model.dropout = args.dropout if args.dropout is not None else config.model.dropout
        config.model.d_model = args.d_model if args.d_model is not None else config.model.d_model
        config.model.pretrained_n_layers = args.n_layers if args.n_layers is not None else config.model.pretrained_n_layers
        config.model.temporal_pool = args.temporal_pool if args.temporal_pool is not None else config.model.temporal_pool
        if args.num_workers is not None:
            config.training.num_workers = args.num_workers
    else:
        config = AttentionConfig()
        config.data.data_dir = args.data or "data/processed"
        config.output.output_dir = args.output or "artifacts/pretrain_aamp"
        config.training.epochs = args.epochs if args.epochs is not None else config.training.epochs
        config.training.batch_size = args.batch_size if args.batch_size is not None else config.training.batch_size
        config.training.learning_rate = args.lr if args.lr is not None else config.training.learning_rate
        config.training.weight_decay = args.weight_decay if args.weight_decay is not None else config.training.weight_decay
        config.model.dropout = args.dropout if args.dropout is not None else config.model.dropout
        config.model.d_model = args.d_model if args.d_model is not None else config.model.d_model
        config.model.pretrained_n_layers = args.n_layers if args.n_layers is not None else config.model.pretrained_n_layers
        config.model.temporal_pool = args.temporal_pool if args.temporal_pool is not None else config.model.temporal_pool
        if args.num_workers is not None:
            config.training.num_workers = args.num_workers

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 --device cuda，但当前 PyTorch 没有可用 CUDA")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"设备: {device}")

    # 输出目录
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_heads = config.model.pretrained_n_heads
    if config.model.d_model % n_heads != 0:
        n_heads = 3 if config.model.d_model % 3 == 0 else 1

    # 构建模型
    model = MiniNeurIPT(
        d_model=config.model.d_model,
        n_heads=n_heads,
        d_ff=config.model.pretrained_d_ff,
        n_layers=config.model.pretrained_n_layers,
        n_channels=config.data.n_channels,
        channel_positions=config.data.channel_positions,
        max_time_steps=config.data.window_samples,
        dropout=max(config.model.dropout, 0.1),
        mask_ratio_range=config.aamp.mask_ratio_range,
        mask_token_ratio=config.aamp.mask_token_ratio,
        random_token_ratio=config.aamp.random_token_ratio,
        unchanged_ratio=config.aamp.unchanged_ratio,
        percentile_low=config.aamp.percentile_low,
        percentile_high=config.aamp.percentile_high,
        amplitude_type=config.aamp.amplitude_type,
        temporal_pool=config.model.temporal_pool,
        pretrain_masking=config.pretraining.masking,
        sampling_rate=config.data.sampling_rate,
        frequency_bands={
            name: tuple(bounds)
            for name, bounds in config.pretraining.bands.items()
        } or None,
        mask_bands=config.pretraining.mask_bands,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"迷你版 NeurIPT 参数量: {n_params:,}")

    # 数据：默认只使用 train+val；all 明确表示 transductive test exposure。
    if args.synthetic:
        train_ds = SyntheticEEGDataset(
            n_samples=2000,
            n_channels=config.data.n_channels,
            time_points=config.data.window_samples,
            n_classes=config.model.n_classes,
        )
        val_ds = SyntheticEEGDataset(
            n_samples=200,
            n_channels=config.data.n_channels,
            time_points=config.data.window_samples,
            n_classes=config.model.n_classes,
            seed=43,
        )
    else:
        dataset = EEGDataset(
            config.data.data_dir,
            normalize=config.data.normalize,
            use_train_stats=True,
            clip_std=config.data.normalize_clip_std,
            include_test=args.unlabeled_splits == "all",
            zero_channel_policy=config.data.zero_channel_policy,
        )
        train_ds = UnlabeledWaveformDataset(dataset.train_dataset)
        val_ds = UnlabeledWaveformDataset(dataset.val_dataset)
        if args.unlabeled_splits == "all":
            train_ds = ConcatDataset([
                train_ds,
                UnlabeledWaveformDataset(dataset.val_dataset),
                UnlabeledWaveformDataset(dataset.test_dataset),
            ])
            protocol_name = "transductive_all_unlabeled"
        elif args.unlabeled_splits == "train_val":
            train_ds = ConcatDataset([train_ds, UnlabeledWaveformDataset(dataset.val_dataset)])
            protocol_name = "development_train_val_unlabeled"
        else:
            protocol_name = "development_train_only_unlabeled"

        # Validation is kept on the original validation subjects for checkpoint selection.
        # For the transductive protocol this is diagnostic only because validation data
        # also appears in the unlabeled training pool.
        val_ds = UnlabeledWaveformDataset(dataset.val_dataset)

    if args.synthetic:
        protocol_name = "synthetic"

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("AAMP 预训练的 train 或 validation 数据为空")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and device.type == "cuda",
        persistent_workers=config.training.persistent_workers and config.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and device.type == "cuda",
        persistent_workers=config.training.persistent_workers and config.training.num_workers > 0,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    config.save_yaml(output_dir / "pretrain_config.yaml")
    metadata = {
        "protocol": protocol_name,
        "unlabeled_splits": args.unlabeled_splits if not args.synthetic else "synthetic",
        "test_exposed_to_pretraining": bool(
            not args.synthetic and args.unlabeled_splits == "all"
        ),
        "validation_is_used_for_checkpoint_selection": (
            args.checkpoint_selection == "validation"
        ),
        "checkpoint_selection": args.checkpoint_selection,
        "data_dir": str(config.data.data_dir),
        "normalization": {
            "enabled": bool(config.data.normalize),
            "mode": "train_subjects_only",
            "clip_std": config.data.normalize_clip_std,
            "zero_channel_policy": config.data.zero_channel_policy,
        },
        "seed": args.seed,
        "device": str(device),
        "n_train_unlabeled": len(train_ds),
        "n_val_unlabeled": len(val_ds),
        "model": model.get_model_info(),
        "aamp": {
            "mask_ratio_range": config.aamp.mask_ratio_range,
            "mask_token_ratio": config.aamp.mask_token_ratio,
            "random_token_ratio": config.aamp.random_token_ratio,
            "unchanged_ratio": config.aamp.unchanged_ratio,
            "percentile_low": config.aamp.percentile_low,
            "percentile_high": config.aamp.percentile_high,
            "amplitude_type": config.aamp.amplitude_type,
            "validation_mask_seed": args.seed + 100000,
        },
        "pretraining": {
            "masking": config.pretraining.masking,
            "sampling_rate": config.data.sampling_rate,
            "bands": config.pretraining.bands,
            "mask_bands": config.pretraining.mask_bands,
            "loss_domain": "complex_rfft_magnitude_error_on_masked_bins",
        },
    }
    if not args.synthetic:
        metadata["subjects"] = {
            "train": sorted(np.unique(dataset.train_subjects).tolist()),
            "val": sorted(np.unique(dataset.val_subjects).tolist()),
            "test": (
                sorted(np.unique(dataset.test_subjects).tolist())
                if hasattr(dataset, "test_subjects") and dataset.test_subjects is not None
                else []
            ),
        }
    with open(output_dir / "pretrain_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    # 训练循环
    best_val_loss = float("inf")
    patience = args.patience
    patience_counter = 0
    history = []

    print(f"\n开始 {config.pretraining.masking} 频域预训练，共 {config.training.epochs} 轮")
    print("=" * 60)

    for epoch in range(config.training.epochs):
        train_loss = pretrain_one_epoch(
            model, train_loader, optimizer, device, scaler, use_amp
        )
        val_loss = _stable_mask_evaluation(
            model, val_loader, device, use_amp, seed=args.seed + 100000
        )

        print(f"Epoch {epoch+1}/{config.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "diagnostic_best_pretrain_model.pt")
            if args.checkpoint_selection == "validation":
                torch.save(model.state_dict(), output_dir / "best_pretrain_model.pt")
            print(f"  记录最低诊断 Val Loss: {best_val_loss:.4f}")
        elif args.checkpoint_selection == "validation":
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，连续 {patience} 轮无改善")
                break

    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "final_pretrain_model.pt")
    with open(output_dir / "pretrain_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    for checkpoint_name in ("best_pretrain_model.pt", "final_pretrain_model.pt"):
        checkpoint_path = output_dir / checkpoint_name
        if checkpoint_path.exists():
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            metadata.setdefault("checkpoint_sha256", {})[checkpoint_name] = digest
    with open(output_dir / "pretrain_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(f"\n预训练完成！最佳 Val Loss: {best_val_loss:.4f}")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
