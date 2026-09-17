"""
AAMP (Amplitude-Aware Masked Pretraining) 振幅感知掩码模块
借鉴 NeurIPS 2025 NeurIPT 的核心创新：

传统 BERT/MAE 的随机掩码对 EEG 太简单——EEG 信号连续平滑，
随机掩码一段后模型只需要局部插值就能重建，学不到真正的脑电特征。

AAMP 的核心思想：按信号振幅（能量）掩码，而不是随机掩码。
- 对每个通道，按振幅从大到小排序
- 随机采样一个百分位点 c ~ U(0,1)
- 以排序后第 c·T 个点为中心，掩码 T·P 个点
- 这样掩码的是"振幅居中"的片段——既不是瞬态尖峰，也不是噪声平坦区
- 逼模型理解脑电的时频结构和节律，而不是简单的局部插值

同时采用 BERT 式的 80/10/10 掩码策略：
- 80% 替换为 [mask] token
- 10% 替换为随机 embedding
- 10% 保持不变
"""
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict


class AAMPMasking:
    """
    AAMP 振幅感知掩码生成器

    Args:
        mask_ratio_range: 掩码比例范围，从列表中随机选择，如 [0.2, 0.35, 0.5]
                          参考 NeurIPT 用动态比例 [20, 35, 50]
        mask_token_ratio: 被掩码点中替换为 [mask] 的比例（默认0.8）
        random_token_ratio: 被掩码点中替换为随机值的比例（默认0.1）
        unchanged_ratio: 被掩码点中保持不变的比例（默认0.1）
        percentile_low: 百分位采样下限（默认0.0）
        percentile_high: 百分位采样上限（默认1.0）
        mask_value: [mask] token 的值（默认0.0）
        amplitude_type: 幅值计算方式
            - "abs": 按绝对值排序（EEG交流信号推荐，正负对称）
            - "raw": 按原始信号值排序（严格复现论文字面表述）
    """

    def __init__(
        self,
        mask_ratio_range: Optional[List[float]] = None,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        unchanged_ratio: float = 0.1,
        percentile_low: float = 0.0,
        percentile_high: float = 1.0,
        mask_value: float = 0.0,
        amplitude_type: str = "abs",
    ):
        if mask_ratio_range is None:
            mask_ratio_range = [0.2, 0.35, 0.5]

        assert abs(mask_token_ratio + random_token_ratio + unchanged_ratio - 1.0) < 1e-6, \
            f"掩码策略比例之和必须为1，当前为 {mask_token_ratio + random_token_ratio + unchanged_ratio}"
        assert amplitude_type in ("abs", "raw"), f"amplitude_type 必须是 'abs' 或 'raw'，当前为 {amplitude_type}"

        self.mask_ratio_range = mask_ratio_range
        self.mask_token_ratio = mask_token_ratio
        self.random_token_ratio = random_token_ratio
        self.unchanged_ratio = unchanged_ratio
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.mask_value = mask_value
        self.amplitude_type = amplitude_type

    def __call__(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 EEG 信号施加 AAMP 掩码

        Args:
            x: EEG 信号，形状 [batch, n_channels, time]
            return_details: 是否返回详细信息（用于调试/可视化）

        Returns:
            masked_x: 掩码后的信号，形状同 x
            mask: 掩码标记（1=被掩码，0=未掩码），形状同 x
            details (可选): 详细信息字典
        """
        batch, n_channels, time = x.shape
        device = x.device
        dtype = x.dtype

        # 随机选择本次的掩码比例
        mask_ratio = np.random.choice(self.mask_ratio_range)
        n_mask = int(time * mask_ratio)

        # 初始化掩码标记
        mask = torch.zeros(batch, n_channels, time, dtype=torch.bool, device=device)

        # 对每个 batch、每个通道单独做 AAMP
        for b in range(batch):
            for c in range(n_channels):
                signal = x[b, c].cpu().numpy()

                # 步骤1：按振幅从大到小排序
                if self.amplitude_type == "abs":
                    amplitudes = np.abs(signal)  # 绝对值（EEG交流信号推荐）
                else:
                    amplitudes = signal  # 原始值（严格复现论文）
                sorted_indices = np.argsort(amplitudes)[::-1]  # 降序排列的索引

                # 步骤2：随机采样百分位点 c ~ U(percentile_low, percentile_high)
                percentile = np.random.uniform(self.percentile_low, self.percentile_high)
                center_rank = int(percentile * time)

                # 步骤3：以 center_rank 为中心，选择 n_mask 个点
                start_rank = max(0, center_rank - n_mask // 2)
                end_rank = min(time, start_rank + n_mask)
                if end_rank - start_rank < n_mask:
                    start_rank = max(0, end_rank - n_mask)

                # 这些是排序后的位置，需要转回原始时间索引
                masked_sorted_ranks = np.arange(start_rank, end_rank)
                masked_time_indices = sorted_indices[masked_sorted_ranks]

                mask[b, c, masked_time_indices] = True

        # 步骤4：BERT 式 80/10/10 掩码策略
        masked_x = x.clone()

        # 对每个被掩码的点，决定是 [mask]、随机、还是不变
        for b in range(batch):
            for c in range(n_channels):
                masked_indices = mask[b, c].nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    continue

                n = len(masked_indices)
                # 随机打乱，然后按比例分配
                perm = torch.randperm(n)
                n_mask_token = int(n * self.mask_token_ratio)
                n_random = int(n * self.random_token_ratio)
                # 剩下的保持不变

                mask_token_idx = masked_indices[perm[:n_mask_token]]
                random_idx = masked_indices[perm[n_mask_token:n_mask_token + n_random]]
                # unchanged_idx = masked_indices[perm[n_mask_token + n_random:]]

                # 80% 替换为 [mask]
                masked_x[b, c, mask_token_idx] = self.mask_value

                # 10% 替换为随机值（从该通道的信号分布中采样）
                if len(random_idx) > 0:
                    signal_std = x[b, c].std().item()
                    signal_mean = x[b, c].mean().item()
                    random_values = torch.randn(len(random_idx), device=device, dtype=dtype) * signal_std + signal_mean
                    masked_x[b, c, random_idx] = random_values

                # 10% 保持不变（不做任何操作）

        if return_details:
            details = {
                "mask_ratio": mask_ratio,
                "n_masked_points": mask.sum().item(),
                "mask_ratio_actual": mask.float().mean().item(),
                "mask_token_count": int(mask.sum().item() * self.mask_token_ratio),
                "random_token_count": int(mask.sum().item() * self.random_token_ratio),
                "unchanged_count": int(mask.sum().item() * self.unchanged_ratio),
            }
            return masked_x, mask, details

        return masked_x, mask

    def get_mask_ratio(self) -> float:
        """随机获取一个掩码比例"""
        return float(np.random.choice(self.mask_ratio_range))

    @staticmethod
    def visualize_mask(
        original: torch.Tensor,
        masked: torch.Tensor,
        mask: torch.Tensor,
        channel: int = 0,
        batch_idx: int = 0,
    ) -> Dict:
        """
        生成可视化所需的数据（不直接画图，返回数据供外部绘图）

        Args:
            original: 原始信号 [batch, n_channels, time]
            masked: 掩码后信号 [batch, n_channels, time]
            mask: 掩码标记 [batch, n_channels, time]
            channel: 要可视化的通道
            batch_idx: batch 索引

        Returns:
            可视化数据字典
        """
        time = original.shape[-1]
        return {
            "time_axis": np.arange(time),
            "original": original[batch_idx, channel].cpu().numpy(),
            "masked": masked[batch_idx, channel].cpu().numpy(),
            "mask": mask[batch_idx, channel].cpu().numpy(),
            "masked_points_count": mask[batch_idx, channel].sum().item(),
        }


def compare_random_vs_aamp(
    x: torch.Tensor,
    mask_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对比随机掩码和 AAMP 掩码（用于消融实验）

    Args:
        x: EEG 信号 [batch, n_channels, time]
        mask_ratio: 掩码比例

    Returns:
        random_masked, random_mask, aamp_masked, aamp_mask
    """
    batch, n_channels, time = x.shape
    device = x.device

    # 随机掩码（BERT 风格）
    random_mask = torch.rand(batch, n_channels, time, device=device) < mask_ratio
    random_masked = x.clone()
    random_masked[random_mask] = 0.0

    # AAMP 掩码
    aamp = AAMPMasking(mask_ratio_range=[mask_ratio])
    aamp_masked, aamp_mask = aamp(x)

    return random_masked, random_mask, aamp_masked, aamp_mask
