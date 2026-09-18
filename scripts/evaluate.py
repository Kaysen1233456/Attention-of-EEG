"""
评估入口脚本

用法：
    python scripts/evaluate.py --model-path artifacts/exp1/best_model.pt --data data/processed
    python scripts/evaluate.py --model-path artifacts/exp1/best_model.pt --synthetic

评估内容：
- 窗口级指标（Accuracy, Balanced Accuracy, Macro F1, ROC-AUC, Confusion Matrix）
- 被试级指标（聚合每个被试的所有窗口预测）
- 每个被试的详细结果
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import json
import numpy as np
from attention_model.config import AttentionConfig
from attention_model.data import EEGDataset, create_synthetic_dataloaders
from attention_model.evaluation import evaluate_model, compute_subject_level_accuracy
from scripts.train import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--model-path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_model_for_evaluation(config: AttentionConfig):
    """Build an evaluation model with the same architecture switches as training."""
    return build_model(config)


def main():
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备: {device}")

    # 加载配置
    if args.config:
        config = AttentionConfig.from_yaml(args.config)
    else:
        config = AttentionConfig()

    model = build_model_for_evaluation(config)

    # 加载权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"模型加载完成: {args.model_path}")

    # 加载数据
    if args.synthetic:
        loaders = create_synthetic_dataloaders(batch_size=32)
    else:
        dataset = EEGDataset(args.data, normalize=config.data.normalize, clip_std=config.data.normalize_clip_std)
        loaders = dataset.get_dataloaders(batch_size=32)

    # 评估
    print("\n在测试集上评估...")
    metrics = evaluate_model(
        model=model,
        loader=loaders["test"],
        device=device,
        n_classes=config.model.n_classes,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"窗口级 Accuracy:        {metrics['accuracy']:.4f}")
    print(f"窗口级 Balanced Acc:    {metrics['balanced_accuracy']:.4f}")
    print(f"窗口级 Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"窗口级 ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"混淆矩阵:\n{np.array(metrics['confusion_matrix'])}")

    if "subject_level_accuracy" in metrics:
        print(f"\n被试级 Accuracy:        {metrics['subject_level_accuracy']:.4f}")
        print(f"被试级 Balanced Acc:    {metrics['subject_level_balanced_accuracy']:.4f}")
        print(f"被试级正确数:            {metrics['subject_level_correct']}/{metrics['subject_level_n_subjects']}")
        print(f"最优阈值:                {metrics['subject_level_best_threshold']:.4f}")
        print(f"最优阈值下 Accuracy:     {metrics['subject_level_best_threshold_accuracy']:.4f}")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
