"""频域掩码预训练：遮频带，在频域算重建损失"""
import torch
import torch.fft as fft
import numpy as np
from typing import Tuple, Optional, Dict


class FrequencyMasking:
    def __init__(self, sfreq=250, n_time=500, bands=None, mask_bands=None):
        self.sfreq = sfreq
        self.n_time = n_time
        self.bands = bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
        }
        self.mask_bands = mask_bands
    
    def __call__(self, x, return_details=False):
        batch, n_channels, time = x.shape
        x_fft = fft.rfft(x, dim=-1)
        n_freq = x_fft.shape[-1]
        freqs = torch.fft.rfftfreq(time, d=1.0/self.sfreq).to(x.device)
        
        # Use a single named band when configured, otherwise sample one or two.
        if self.mask_bands is not None:
            unknown = set(self.mask_bands) - set(self.bands)
            if unknown:
                raise ValueError(f"Unknown frequency bands: {sorted(unknown)}")
            chosen = list(self.mask_bands)
        else:
            n_mask = int(np.random.choice([1, 2]))
            band_names = list(self.bands.keys())
            chosen = np.random.choice(band_names, size=n_mask, replace=False).tolist()
        
        # 频域mask：哪些频点被遮了
        freq_mask = torch.zeros(n_freq, dtype=torch.bool, device=x.device)
        for name in chosen:
            low, high = self.bands[name]
            freq_mask |= ((freqs >= low) & (freqs < high))
        
        # Return the masked spectrum directly so the pretraining encoder sees
        # the frequency-domain corruption and the loss uses the same domain.
        x_fft_masked = x_fft.masked_fill(freq_mask, 0.0)
        
        # 返回：给模型的输入 + 频域mask标记（用于算loss）
        if return_details:
            return x_fft_masked, freq_mask, {'masked_bands': list(chosen)}
        return x_fft_masked, freq_mask
