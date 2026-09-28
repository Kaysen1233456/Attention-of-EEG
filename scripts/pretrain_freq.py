"""频域掩码预训练：在频域算重建损失"""
import sys, argparse
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from attention_model.models import MiniNeurIPT
from attention_model.data import EEGDataset
from attention_model.data.freq_masking import FrequencyMasking
from attention_model.training.trainer import set_seed


class UnlabeledWaveformDataset(torch.utils.data.Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return {"waveform": item["waveform"]}


CHANNEL_POSITIONS = [
    [0.0,1.0,0.3],[-0.4,0.8,0.5],[0.4,0.8,0.5],
    [-1.0,0.0,0.0],[1.0,0.0,0.0],[0.0,-1.0,0.3],
    [-0.4,-0.8,0.5],[0.4,-0.8,0.5],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--lr", type=float, default=0.000503)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/pretrain_freq")
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)
    
    dataset = EEGDataset(args.data or "data/processed", normalize=True,
                         use_train_stats=True, clip_std=8.0, include_test=False)
    train_ds = ConcatDataset([UnlabeledWaveformDataset(dataset.train_dataset),
                              UnlabeledWaveformDataset(dataset.val_dataset)])
    val_ds = UnlabeledWaveformDataset(dataset.val_dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}", flush=True)
    
    model = MiniNeurIPT(d_model=96, n_heads=8, d_ff=384, n_layers=4,
                         n_channels=8, channel_positions=CHANNEL_POSITIONS, dropout=0.15)
    model = model.to(device)
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    
    freq_masker = FrequencyMasking(sfreq=250, n_time=500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    best_val = float('inf'); pat = 0
    
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0; nb = 0
        for batch in train_loader:
            x = batch["waveform"].to(device)
            masked_x, freq_mask = freq_masker(x)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=True):
                encoded = model.encode(masked_x)
                bs, nc, et, dm = encoded.shape
                decoded = model.decoder(encoded.reshape(-1, dm))
                recon = decoded.reshape(bs, nc, et)
                if et != 500:
                    recon = F.interpolate(recon.reshape(bs*nc,1,et), size=500,
                                          mode='linear', align_corners=False).reshape(bs,nc,500)
                # 频域loss：比较FFT后的频谱
                recon_fft = torch.fft.rfft(recon, dim=-1)
                orig_fft = torch.fft.rfft(x, dim=-1)
                # 只在被遮的频点上算loss
                loss = F.l1_loss(recon_fft[..., freq_mask], orig_fft[..., freq_mask])
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item(); nb += 1
        tr_loss /= nb
        
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["waveform"].to(device)
                masked_x, freq_mask = freq_masker(x)
                encoded = model.encode(masked_x)
                bs, nc, et, dm = encoded.shape
                decoded = model.decoder(encoded.reshape(-1, dm))
                recon = decoded.reshape(bs, nc, et)
                if et != 500:
                    recon = F.interpolate(recon.reshape(bs*nc,1,et), size=500,
                                          mode='linear', align_corners=False).reshape(bs,nc,500)
                recon_fft = torch.fft.rfft(recon, dim=-1)
                orig_fft = torch.fft.rfft(x, dim=-1)
                loss = F.l1_loss(recon_fft[..., freq_mask], orig_fft[..., freq_mask])
                vl += loss.item(); nv += 1
        vl /= nv
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train: {tr_loss:.4f} | Val: {vl:.4f}", flush=True)
        if vl < best_val:
            best_val = vl; pat = 0
            torch.save(model.state_dict(), out / "best_pretrain_model.pt")
            print(f"  -> 最佳: {best_val:.4f}", flush=True)
        else:
            pat += 1
            if pat >= 5:
                print(f"早停 epoch {epoch+1}", flush=True); break
    
    print(f"完成! 最佳Val Loss: {best_val:.4f}", flush=True)

if __name__ == "__main__":
    main()
