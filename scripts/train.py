"""
训练入口脚本

用法：
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --data data/processed --output artifacts/exp1 --lr 0.001 --batch-size 32

支持：
- 从 YAML 配置文件加载
- 命令行参数覆盖
- 多种子训练
- 自动保存配置和结果
"""
import sys
import os
import argparse
import subprocess
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from attention_model.config import AttentionConfig
from attention_model.models import DualBranchAttentionClassifier, SingleBranchAttentionClassifier, MiniNeurIPT, MiniNeurIPTClassifier
from attention_model.data import EEGDataset, create_synthetic_dataloaders
from attention_model.training import AttentionTrainer, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="训练注意力解码模型")
    parser.add_argument("--fold-manifest", default=None)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--architecture", type=str, default=None, choices=["dual_branch", "single_branch", "mini_neuript", "mini_neuript_classifier"])
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="batch大小")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--d-model", type=int, default=None, help="模型隐藏维度")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader工作进程数")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="随机种子列表")
    parser.add_argument("--multi-seed", action="store_true", help="多种子训练")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据（用于冒烟测试）")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--pretrained-path", type=str, default=None, help="AAMP预训练MiniNeurIPT权重")
    parser.add_argument("--ablation", type=str, default=None, choices=["B0", "B1", "B2", "B3", "B4", "B5"])
    parser.add_argument("--evaluate-test", action="store_true", help="训练完成后评估测试集；默认不访问测试集")
    return parser.parse_args()


def apply_ablation_variant(config: AttentionConfig, variant: str) -> None:
    """Apply one controlled architecture increment for B0-B5."""
    if variant == "B0":
        config.model.architecture = "dual_branch"
        config.model.d_model = 33
        config.model.branch_conv1_out = 32
        config.model.branch_conv1_kernel = 5
        config.model.branch_conv1_stride = 2
        config.model.branch_conv2_out = 64
        config.model.branch_conv2_kernel = 3
        config.model.branch_conv2_stride = 2
        config.model.fusion_hidden = 64
        config.model.fusion_use_diff = True
        config.model.fusion_use_product = True
        config.model.n_classes = 2
        config.model.dropout = 0.0
        config.model.use_3d_embedding = True
        config.model.channel_reduce = "none"
        config.model.use_multilayer_concat = True
        config.model.use_iilp_pooling = True
        config.model.pretrained_backbone_path = None
        config.model.freeze_pretrained_backbone = False
        return
    if variant not in {"B1", "B2", "B3", "B4", "B5"}:
        raise ValueError(f"Unknown ablation variant: {variant}")
    config.model.architecture = "mini_neuript_classifier"
    config.model.use_pmoe = variant in {"B4", "B5"}
    config.model.use_iilp_pooling = variant in {"B2", "B3", "B4", "B5"}
    config.model.use_difference_feature = variant in {"B3", "B4", "B5"}
    config.model.use_product_feature = variant in {"B3", "B4", "B5"}
    config.model.classifier_type = "swiglu" if variant == "B5" else "mlp"


def build_model(config: AttentionConfig, pretrained_path=None):
    """根据配置构建模型"""
    if config.model.architecture == "dual_branch":
        model = DualBranchAttentionClassifier(
            n_channels=config.data.n_channels,
            left_indices=config.data.left_channel_indices,
            right_indices=config.data.right_channel_indices,
            d_model=config.model.d_model,
            conv1_out=config.model.branch_conv1_out,
            conv1_kernel=config.model.branch_conv1_kernel,
            conv1_stride=config.model.branch_conv1_stride,
            conv2_out=config.model.branch_conv2_out,
            conv2_kernel=config.model.branch_conv2_kernel,
            conv2_stride=config.model.branch_conv2_stride,
            fusion_hidden=config.model.fusion_hidden,
            n_classes=config.model.n_classes,
            activation=config.model.activation,
            dropout=config.model.dropout,
            use_3d_embedding=config.model.use_3d_embedding,
            channel_reduce=config.model.channel_reduce,
            use_multilayer_concat=config.model.use_multilayer_concat,
            use_diff_feature=config.model.fusion_use_diff,
            use_product_feature=config.model.fusion_use_product,
            channel_positions=config.data.channel_positions,
            use_iilp_pooling=config.model.use_iilp_pooling,
            pretrained_path=pretrained_path or config.model.pretrained_backbone_path,
            pretrained_n_heads=config.model.pretrained_n_heads,
            pretrained_n_layers=config.model.pretrained_n_layers,
            pretrained_d_ff=config.model.pretrained_d_ff,
            freeze_pretrained_backbone=config.model.freeze_pretrained_backbone,
        )
    elif config.model.architecture == "single_branch":
        model = SingleBranchAttentionClassifier(
            n_channels=config.data.n_channels,
            d_model=config.model.d_model,
            conv1_out=config.model.branch_conv1_out,
            conv1_kernel=config.model.branch_conv1_kernel,
            conv1_stride=config.model.branch_conv1_stride,
            conv2_out=config.model.branch_conv2_out,
            conv2_kernel=config.model.branch_conv2_kernel,
            conv2_stride=config.model.branch_conv2_stride,
            hidden_dim=config.model.fusion_hidden,
            n_classes=config.model.n_classes,
            activation=config.model.activation,
            dropout=config.model.dropout,
            use_3d_embedding=config.model.use_3d_embedding,
            channel_reduce=config.model.channel_reduce,
            use_multilayer_concat=config.model.use_multilayer_concat,
            channel_positions=config.data.channel_positions,
        )
    elif config.model.architecture in ("mini_neuript", "mini_neuript_classifier"):
        n_heads = config.model.pretrained_n_heads
        if config.model.d_model % n_heads != 0:
            n_heads = 3 if config.model.d_model % 3 == 0 else 1
        model = MiniNeurIPTClassifier(
            d_model=config.model.d_model,
            n_heads=n_heads,
            d_ff=config.model.pretrained_d_ff,
            n_layers=config.model.pretrained_n_layers,
            n_channels=config.data.n_channels,
            n_classes=config.model.n_classes,
            channel_positions=config.data.channel_positions,
            left_indices=config.data.left_channel_indices,
            right_indices=config.data.right_channel_indices,
            dropout=max(config.model.dropout, 0.1),
            pretrained_path=pretrained_path or config.model.pretrained_backbone_path,
            freeze_encoder=config.model.freeze_pretrained_backbone,
            temporal_pool=config.model.temporal_pool,
            temporal_frontend=config.model.temporal_frontend,
            temporal_kernel=config.model.temporal_kernel,
            iilp_pooling=config.model.iilp_pooling,
            iilp_attention_dropout=config.model.iilp_attention_dropout,
            use_iilp_pooling=config.model.use_iilp_pooling,
            use_pmoe=config.model.use_pmoe,
            classifier_type=config.model.classifier_type,
            use_difference_feature=config.model.use_difference_feature,
            use_product_feature=config.model.use_product_feature,
        )
    else:
        raise ValueError(f"未知的架构: {config.model.architecture}")

    return model


def main():
    args = parse_args()
    if args.multi_seed:
        if args.evaluate_test:
            raise ValueError("Multi-seed development runs cannot evaluate test")
        base = AttentionConfig.from_yaml(args.config) if args.config else AttentionConfig()
        seeds = args.seeds or base.training.seeds
        root = Path(args.output or base.output.output_dir)
        rows = []
        for seed in seeds:
            target = root / f"seed_{seed}"
            if target.exists():
                raise FileExistsError(target)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--seeds",
                str(seed),
                "--output",
                str(target),
            ]
            for flag, value in (
                ("--config", args.config),
                ("--data", args.data),
                ("--architecture", args.architecture),
                ("--lr", args.lr),
                ("--batch-size", args.batch_size),
                ("--epochs", args.epochs),
                ("--d-model", args.d_model),
                ("--num-workers", args.num_workers),
                ("--device", args.device),
                ("--pretrained-path", args.pretrained_path),
                ("--ablation", args.ablation),
                ("--fold-manifest", args.fold_manifest),
                ("--fold-index", args.fold_index if args.fold_manifest else None),
            ):
                if value is not None:
                    command.extend([flag, str(value)])
            if args.synthetic:
                command.append("--synthetic")
            subprocess.run(command, check=True)
            result = json.loads((target / "results.json").read_text())
            rows.append({
                "seed": seed,
                "metrics": result["best_val_metrics"],
                "best_epoch": result["best_epoch"],
                "checkpoint": str(target / "best_model.pt"),
            })
        keys = ("balanced_accuracy", "macro_f1", "roc_auc")
        summary = {"split": "validation", "per_seed": rows,
                   "mean": {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys},
                   "std": {key: float(np.std([row["metrics"][key] for row in rows])) for key in keys}}
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # 加载配置
    if args.config:
        config = AttentionConfig.from_yaml(args.config)
        print(f"从配置文件加载: {args.config}")
    else:
        config = AttentionConfig()
        print("使用默认配置")
    if args.ablation:
        apply_ablation_variant(config, args.ablation)

    # 命令行参数覆盖
    if args.data:
        config.data.data_dir = args.data
    if args.output:
        config.output.output_dir = args.output
    if args.architecture:
        config.model.architecture = args.architecture
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.d_model:
        config.model.d_model = args.d_model
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.seeds:
        config.training.seeds = args.seeds

    seed = config.training.seeds[0]
    config.random_seed = seed
    set_seed(seed)

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")

    # 构建模型
    model = build_model(config, args.pretrained_path)
    if args.pretrained_path:
        config.model.pretrained_backbone_path = args.pretrained_path
        config.model.use_pretrained_backbone = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型架构: {config.model.architecture}")
    print(f"模型参数量: {n_params:,}")

    # 加载数据
    if args.synthetic:
        print("使用合成数据（冒烟测试）")
        loaders = create_synthetic_dataloaders(
            batch_size=config.training.batch_size,
            n_channels=config.data.n_channels,
            time_points=config.data.window_samples,
            n_classes=config.model.n_classes,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.persistent_workers,
        )
    else:
        print(f"加载数据: {config.data.data_dir}")
        fold = None
        if args.fold_manifest:
            manifest_path = Path(args.fold_manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            folds = manifest.get("folds")
            if not isinstance(folds, list) or not folds:
                raise ValueError("fold manifest must contain a non-empty 'folds' list")
            if not 0 <= args.fold_index < len(folds):
                raise IndexError(
                    f"fold-index {args.fold_index} out of range for {len(folds)} folds"
                )
            fold = folds[args.fold_index]
        dataset = EEGDataset(
            data_dir=config.data.data_dir,
            normalize=config.data.normalize,
            clip_std=config.data.normalize_clip_std,
            include_test=args.evaluate_test,
            fold=fold,
            normalization_mode=config.data.normalization_mode,
            zero_channel_policy=config.data.zero_channel_policy,
        )
        loaders = dataset.get_dataloaders(
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory and device.type == "cuda",
            persistent_workers=config.training.persistent_workers,
        )
        print(f"训练集: {len(loaders['train'].dataset)} 样本")
        print(f"验证集: {len(loaders['val'].dataset)} 样本")
        if args.evaluate_test:
            print(f"测试集: {len(loaders['test'].dataset)} 样本")

    # 创建训练器
    trainer = AttentionTrainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"] if args.evaluate_test else None,
        device=device,
    )

    # 训练
    if args.multi_seed:
        print("多种子训练模式")
        results = trainer.train_multi_seed()
    else:
        results = trainer.train()

    print("\n训练完成！")
    print(f"结果保存在: {config.output.output_dir}")

    return results


if __name__ == "__main__":
    main()
