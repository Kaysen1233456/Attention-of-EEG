"""在验证集上搜索最优阈值，然后在测试集上评估"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from attention_model.models import MiniNeurIPTClassifier
from attention_model.training.trainer import set_seed

CHANNEL_POSITIONS = [
    [0.0, 1.0, 0.3], [-0.4, 0.8, 0.5], [0.4, 0.8, 0.5],
    [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.3],
    [-0.4, -0.8, 0.5], [0.4, -0.8, 0.5],
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = "data/processed/mental_arithmetic_250hz_8ch"
    val_x = np.load(f"{DATA_DIR}/val_waveforms.npy").astype(np.float32)
    val_y = np.load(f"{DATA_DIR}/val_labels.npy")
    test_x = np.load(f"{DATA_DIR}/test_waveforms.npy").astype(np.float32)
    test_y = np.load(f"{DATA_DIR}/test_labels.npy")

    val_loader = DataLoader(TensorDataset(torch.tensor(val_x), torch.tensor(val_y, dtype=torch.long)), batch_size=42)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long)), batch_size=42)

    model = MiniNeurIPTClassifier(
        d_model=96, n_heads=8, d_ff=384, n_layers=4,
        n_channels=8, n_classes=2,
        channel_positions=CHANNEL_POSITIONS,
        left_indices=[0,1,2,3,4], right_indices=[5,6,7],
        dropout=0.395, pretrained_path="artifacts/pretrain_aamp_best_params_seed43/diagnostic_best_pretrain_model.pt",
        freeze_encoder=True, temporal_pool=4,
        iilp_pooling="attention", use_iilp_pooling=True,
        use_pmoe=True, classifier_type="swiglu",
        use_difference_feature=True, use_product_feature=True,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()

    val_probs, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb.to(device))
            val_probs.extend(torch.softmax(out["logits"], 1)[:, 1].cpu().numpy())
            val_labels.extend(yb.numpy())
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    test_probs, test_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            test_probs.extend(torch.softmax(out["logits"], 1)[:, 1].cpu().numpy())
            test_labels.extend(yb.numpy())
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)

    best_thresh = 0.5
    best_val_balacc = 0
    for t in np.arange(0.1, 0.9, 0.01):
        balacc = balanced_accuracy_score(val_labels, (val_probs > t).astype(int))
        if balacc > best_val_balacc:
            best_val_balacc = balacc
            best_thresh = t

    test_pred = (test_probs > best_thresh).astype(int)
    test_balacc = balanced_accuracy_score(test_labels, test_pred)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_pred)

    print(f"Seed {args.seed}")
    print(f"  验证集最优阈值: {best_thresh:.2f}, Val BalAcc: {best_val_balacc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test BalAcc: {test_balacc:.4f}")
    print(f"  Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
