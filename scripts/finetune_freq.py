"""用频域掩码预训练权重微调，参数和Gen4完全一致"""
import sys, argparse
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from attention_model.models import MiniNeurIPTClassifier
from attention_model.training.trainer import set_seed

CHANNEL_POSITIONS = [
    [0.0, 1.0, 0.3], [-0.4, 0.8, 0.5], [0.4, 0.8, 0.5],
    [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.3],
    [-0.4, -0.8, 0.5], [0.4, -0.8, 0.5],
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/finetune_freq")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DATA_DIR = "data/processed/mental_arithmetic_250hz_8ch"
    train_x = np.load(f"{DATA_DIR}/train_waveforms.npy").astype(np.float32)
    train_y = np.load(f"{DATA_DIR}/train_labels.npy")
    val_x = np.load(f"{DATA_DIR}/val_waveforms.npy").astype(np.float32)
    val_y = np.load(f"{DATA_DIR}/val_labels.npy")
    test_x = np.load(f"{DATA_DIR}/test_waveforms.npy").astype(np.float32)
    test_y = np.load(f"{DATA_DIR}/test_labels.npy")
    
    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_x), torch.tensor(val_y, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=42, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=42)
    test_loader = DataLoader(test_ds, batch_size=42)
    
    # Gen4配置
    model = MiniNeurIPTClassifier(
        d_model=48, n_heads=6, d_ff=384, n_layers=2,
        n_channels=8, n_classes=2,
        channel_positions=CHANNEL_POSITIONS,
        left_indices=[0,1,2,3,4], right_indices=[5,6,7],
        dropout=0.395, pretrained_path=args.pretrained,
        freeze_encoder=True, temporal_pool=4,
        iilp_pooling="attention", use_iilp_pooling=True,
        use_pmoe=True, classifier_type="swiglu",
        use_difference_feature=True, use_product_feature=False,
    )
    model = model.to(device)
    
    # Gen4训练参数
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.23e-5, weight_decay=5.29e-6
    )
    class_weights = torch.tensor([1.13, 2.55]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_auc = 0
    patience = 0
    best_state = None
    
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out["logits"], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device))
                vp.append(out["logits"][:,1].cpu().numpy())
                vl.append(yb.numpy())
        vp = np.concatenate(vp); vl = np.concatenate(vl)
        va = roc_auc_score(vl, vp)
        
        if va > best_val_auc:
            best_val_auc = va
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    
    # 测试集
    model.load_state_dict(best_state)
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            tp.append(out["logits"][:,1].cpu().numpy())
            tl.append(yb.numpy())
    tp = np.concatenate(tp); tl = np.concatenate(tl)
    test_auc = roc_auc_score(tl, tp)
    
    best_t, best_b = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.01):
        b = balanced_accuracy_score(vl, (vp > t).astype(int))
        if b > best_b: best_b = b; best_t = t
    test_bal = balanced_accuracy_score(tl, (tp > best_t).astype(int))
    test_f1 = f1_score(tl, (tp > best_t).astype(int))
    
    print(f"Seed {args.seed} | ValAUC={best_val_auc:.4f} | TestAUC={test_auc:.4f} | BalAcc={test_bal:.4f} | F1={test_f1:.4f}", flush=True)
    print(f"Gen4基线: AUC=0.6926, BalAcc=0.6704", flush=True)

if __name__ == "__main__":
    main()
