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
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from attention_model.config import AttentionConfig
from attention_model.models import DualBranchAttentionClassifier, SingleBranchAttentionClassifier, MiniNeurIPT
from attention_model.data import EEGDataset, create_synthetic_dataloaders
from attention_model.training import AttentionTrainer, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="训练注意力解码模型")
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--architecture", type=str, default=None, choices=["dual_branch", "mini_neuript"])
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="batch大小")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="随机种子列表")
    parser.add_argument("--multi-seed", action="store_true", help="多种子训练")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据（用于冒烟测试）")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--pretrained-path", type=str, default=None, help="AAMP预训练MiniNeurIPT权重")
    return parser.parse_args()


def build_model(config: AttentionConfig, pretrained_path=None):
    """根据配置构建模型"""
    if config.model.architecture == "dual_branch":
        model = DualBranchAttentionClassifier(
            n_channels=config.data.n_channels,
            left_indices=config.data.left_channel_indices,
            right_indices=config.data.right_channel_indices,
            d_model=config.model.d_model,
            conv1_out=config.model.branch_conv1_out,
            conv2_out=config.model.branch_conv2_out,
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
            conv2_out=config.model.branch_conv2_out,
            hidden_dim=config.model.fusion_hidden,
            n_classes=config.model.n_classes,
            activation=config.model.activation,
            dropout=config.model.dropout,
            use_3d_embedding=config.model.use_3d_embedding,
            channel_reduce=config.model.channel_reduce,
            use_multilayer_concat=config.model.use_multilayer_concat,
            channel_positions=config.data.channel_positions,
        )
    elif config.model.architecture == "mini_neuript":
        model = MiniNeurIPT(
            d_model=config.model.d_model,
            n_heads=8,
            d_ff=config.model.pretrained_d_ff,
            n_layers=4,
            n_channels=config.data.n_channels,
            channel_positions=config.data.channel_positions,
        )
    else:
        raise ValueError(f"未知的架构: {config.model.architecture}")

    return model


def main():
    args = parse_args()

    # 加载配置
    if args.config:
        config = AttentionConfig.from_yaml(args.config)
        print(f"从配置文件加载: {args.config}")
    else:
        config = AttentionConfig()
        print("使用默认配置")

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
    if args.seeds:
        config.training.seeds = args.seeds

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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
        )
    else:
        print(f"加载数据: {config.data.data_dir}")
        dataset = EEGDataset(
            data_dir=config.data.data_dir,
            normalize=config.data.normalize,
        )
        loaders = dataset.get_dataloaders(
            batch_size=config.training.batch_size,
        )
        print(f"训练集: {len(loaders['train'].dataset)} 样本")
        print(f"验证集: {len(loaders['val'].dataset)} 样本")
        print(f"测试集: {len(loaders['test'].dataset)} 样本")

    # 创建训练器
    trainer = AttentionTrainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        device=device,
    )

    # 训练
    if args.multi_seed:
        print("多种子训练模式")
        results = trainer.train_multi_seed()
    else:
        set_seed(config.training.seeds[0])
        results = trainer.train()

    print("\n训练完成！")
    print(f"结果保存在: {config.output.output_dir}")

    return results


if __name__ == "__main__":
    main()
