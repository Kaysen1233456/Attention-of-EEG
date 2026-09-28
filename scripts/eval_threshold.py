"""用验证集选最优阈值，评估测试集"""
import sys, argparse
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from attention_model.data.dataset import EEGDataset
from attention_model.models import MiniNeurIPTClassifier
from attention_model.training.trainer import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载配置
    with open(args.config) as f:
        config = type('Config', (), {})()
        raw = yaml.safe_load(f)
        for k, v in raw.items():
            setattr(config, k, type('Sub', (), v) if isinstance(v, dict) else v)

    # 加载数据（带normalize）
    dataset = EEGDataset(
        data_dir=config.data.data_dir,
        normalize=config.data.normalize,
        clip_std=config.data.normalize_clip_std,
        include_test=True,
        normalization_mode=config.data.normalization_mode,
        zero_channel_policy=config.data.zero_channel_policy,
    )
    loaders = dataset.get_dataloaders(batch_size=42, num_workers=0)

    # 构建模型
    model = MiniNeurIPTClassifier(
        d_model=config.model.d_model,
        n_heads=config.model.pretrained_n_heads,
        d_ff=config.model.pretrained_d_ff,
        n_layers=config.model.pretrained_n_layers,
        n_channels=config.data.n_channels,
        n_classes=config.model.n_classes,
        channel_positions=config.data.channel_positions,
        left_indices=config.data.left_channel_indices,
        right_indices=config.data.right_channel_indices,
        dropout=config.model.dropout,
        pretrained_path=config.model.pretrained_backbone_path,
        freeze_encoder=True,
        temporal_pool=config.model.temporal_pool,
        iilp_pooling="attention",
        use_iilp_pooling=config.model.use_iilp_pooling,
        use_pmoe=config.model.use_pmoe,
        classifier_type=config.model.classifier_type,
        use_difference_feature=config.model.use_difference_feature,
        use_product_feature=config.model.use_product_feature,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()

    # 收集验证集概率
    val_probs, val_labels = [], []
    with torch.no_grad():
        for batch in loaders["val"]:
            x = batch["waveform"].to(device)
            out = model(x)
            val_probs.extend(torch.softmax(out["logits"], 1)[:, 1].cpu().numpy())
            val_labels.extend(batch["label"].numpy())
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    # 收集测试集概率
    test_probs, test_labels = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            x = batch["waveform"].to(device)
            out = model(x)
            test_probs.extend(torch.softmax(out["logits"], 1)[:, 1].cpu().numpy())
            test_labels.extend(batch["label"].numpy())
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)

    # 验证集选最优阈值
    best_thresh, best_val_balacc = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.01):
        balacc = balanced_accuracy_score(val_labels, (val_probs > t).astype(int))
        if balacc > best_val_balacc:
            best_val_balacc = balacc
            best_thresh = t

    # 测试集评估
    test_pred = (test_probs > best_thresh).astype(int)
    test_balacc = balanced_accuracy_score(test_labels, test_pred)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_pred)

    print(f"Seed {args.seed}")
    print(f"  最优阈值: {best_thresh:.2f}, Val BalAcc: {best_val_balacc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}, Test BalAcc: {test_balacc:.4f}, Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
