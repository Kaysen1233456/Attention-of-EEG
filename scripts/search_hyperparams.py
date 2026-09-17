"""
超参数搜索入口脚本

用法：
    python scripts/search_hyperparams.py --method quasi_random --n-trials 50 --output artifacts/search
    python scripts/search_hyperparams.py --method bayesian --n-trials 30 --n-initial 10

支持：
- 准随机搜索（Sobol/Halton/Latin Hypercube）
- 贝叶斯优化（EI/UCB/POI）
- 自动保存搜索结果
- 搜索时减少 epoch 加速
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import json
from attention_model.config import AttentionConfig
from attention_model.models import DualBranchAttentionClassifier
from attention_model.data import EEGDataset, create_synthetic_dataloaders
from attention_model.training import AttentionTrainer, set_seed
from attention_model.search import QuasiRandomSearch, BayesianOptimization


def parse_args():
    parser = argparse.ArgumentParser(description="超参数搜索")
    parser.add_argument("--method", type=str, default="quasi_random", choices=["quasi_random", "bayesian"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-initial", type=int, default=10, help="贝叶斯优化的初始随机试验数")
    parser.add_argument("--quasi-method", type=str, default="sobol", choices=["sobol", "halton", "latin_hypercube"])
    parser.add_argument("--acquisition", type=str, default="ei", choices=["ei", "ucb", "poi"])
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="artifacts/hyperparam_search")
    parser.add_argument("--search-epochs", type=int, default=30, help="搜索时每轮训练的epoch数（减少加速）")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_objective_fn(base_config, loaders, device, search_epochs):
    """创建目标函数（闭包，捕获 base_config 和 loaders）"""
    def objective(params: dict) -> dict:
        # 复制基础配置
        config = AttentionConfig.from_yaml("configs/baseline.yaml") if Path("configs/baseline.yaml").exists() else AttentionConfig()

        # 应用搜索到的参数
        config.update_from_dict(params)
        config.training.epochs = search_epochs
        config.output.output_dir = str(Path(base_config.output.output_dir) / f"trial_{params.get('trial_idx', 'tmp')}")

        # 构建模型
        model = DualBranchAttentionClassifier(
            n_channels=config.data.n_channels,
            d_model=config.model.d_model,
            conv1_out=config.model.branch_conv1_out,
            conv2_out=config.model.branch_conv2_out,
            fusion_hidden=config.model.fusion_hidden,
            n_classes=config.model.n_classes,
            dropout=config.model.dropout,
        )

        # 训练
        set_seed(42)
        trainer = AttentionTrainer(
            model=model,
            config=config,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=None,
            device=device,
        )
        results = trainer.train()

        # 返回验证集指标
        val_metrics = results.get("final_val_metrics", {})
        return val_metrics

    return objective


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 基础配置
    base_config = AttentionConfig()
    base_config.output.output_dir = str(output_dir)

    # 搜索空间
    search_space = {
        "learning_rate": [1e-4, 5e-3, "float"],
        "batch_size": [16, 64, "int"],
        "d_model": [24, 48, 72, 96],
        "branch_conv1_out": [16, 64, "int"],
        "branch_conv2_out": [32, 128, "int"],
        "fusion_hidden": [32, 128, "int"],
        "dropout": [0.0, 0.3, "float"],
    }

    # 加载数据
    if args.synthetic:
        loaders = create_synthetic_dataloaders(batch_size=32)
    else:
        dataset = EEGDataset(args.data)
        loaders = dataset.get_dataloaders(batch_size=32)

    # 创建目标函数
    objective_fn = make_objective_fn(base_config, loaders, device, args.search_epochs)

    # 运行搜索
    if args.method == "quasi_random":
        searcher = QuasiRandomSearch(
            search_space=search_space,
            n_trials=args.n_trials,
            method=args.quasi_method,
            seed=args.seed,
            objective_metric="balanced_accuracy",
            maximize=True,
        )
    else:
        searcher = BayesianOptimization(
            search_space=search_space,
            n_trials=args.n_trials,
            n_initial=args.n_initial,
            acquisition=args.acquisition,
            seed=args.seed,
            objective_metric="balanced_accuracy",
            maximize=True,
        )

    print(f"开始超参数搜索: {args.method}")
    print(f"搜索空间: {search_space}")
    print(f"试验次数: {args.n_trials}")

    summary = searcher.run(objective_fn, output_dir=str(output_dir))

    print("\n" + "=" * 60)
    print("搜索完成！")
    print(f"最佳参数: {summary['best_params']}")
    print(f"最佳指标: {summary['best_value']:.4f}")
    print(f"结果保存在: {output_dir}")

    # 保存最佳参数为 YAML
    best_config = AttentionConfig()
    best_config.update_from_dict(summary["best_params"])
    best_config.save_yaml(output_dir / "best_config.yaml")
    print(f"最佳配置已保存: {output_dir / 'best_config.yaml'}")


if __name__ == "__main__":
    main()
