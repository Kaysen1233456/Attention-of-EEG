"""
AAMP 自监督预训练入口脚本（阶段二）

用法：
    python scripts/pretrain_aamp.py --data data/processed --output artifacts/pretrain
    python scripts/pretrain_aamp.py --config configs/pretrain.yaml

预训练完成后，可以用 train.py 进行微调，或用蒸馏脚本把知识蒸馏到小模型。
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from attention_model.config import AttentionConfig
from attention_model.models import MiniNeurIPT
from attention_model.data import EEGDataset, SyntheticEEGDataset
from attention_model.training.losses import AAMPReconstructionLoss
from attention_model.training.trainer import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="AAMP 自监督预训练")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="artifacts/pretrain_aamp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pretrain_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    """预训练一个 epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        waveforms = batch["waveform"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
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
def evaluate_pretrain(model, loader, criterion, device, use_amp):
    """评估预训练（重建损失）"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        waveforms = batch["waveform"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(waveforms, apply_mask=True)
            loss = output["loss"]
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建模型
    model = MiniNeurIPT(
        d_model=args.d_model,
        n_heads=8,
        d_ff=args.d_model * 4,
        n_layers=args.n_layers,
        n_channels=4,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"迷你版 NeurIPT 参数量: {n_params:,}")

    # 数据
    if args.synthetic:
        train_ds = SyntheticEEGDataset(n_samples=2000, n_channels=4, time_points=500)
        val_ds = SyntheticEEGDataset(n_samples=200, n_channels=4, time_points=500, seed=43)
    else:
        dataset = EEGDataset(args.data)
        train_ds = dataset.train_dataset
        val_ds = dataset.val_dataset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = AAMPReconstructionLoss(loss_type="l1")
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    # 训练循环
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print(f"\n开始 AAMP 预训练，共 {args.epochs} 轮")
    print("=" * 60)

    for epoch in range(args.epochs):
        train_loss = pretrain_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss = evaluate_pretrain(model, val_loader, criterion, device, use_amp)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_pretrain_model.pt")
            print(f"  保存最佳模型，Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，连续 {patience} 轮无改善")
                break

    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "final_pretrain_model.pt")
    print(f"\n预训练完成！最佳 Val Loss: {best_val_loss:.4f}")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
