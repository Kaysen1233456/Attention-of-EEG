"""
AAMP 振幅感知掩码模块单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pytest
from attention_model.data.aamp_masking import AAMPMasking, compare_random_vs_aamp


class TestAAMPMasking:
    """测试 AAMP 掩码"""

    def test_init_default(self):
        """测试默认初始化"""
        aamp = AAMPMasking()
        assert aamp.mask_ratio_range == [0.2, 0.35, 0.5]
        assert aamp.mask_token_ratio == 0.8
        assert aamp.random_token_ratio == 0.1
        assert aamp.unchanged_ratio == 0.1

    def test_mask_ratio_sum(self):
        """测试掩码策略比例之和为1"""
        with pytest.raises(AssertionError):
            AAMPMasking(mask_token_ratio=0.7, random_token_ratio=0.2, unchanged_ratio=0.2)

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        aamp = AAMPMasking(mask_ratio_range=[0.5])
        x = torch.randn(2, 4, 500)
        masked_x, mask = aamp(x)
        assert masked_x.shape == x.shape
        assert mask.shape == x.shape
        assert mask.dtype == torch.bool

    def test_mask_applied(self):
        """测试掩码确实被应用"""
        aamp = AAMPMasking(mask_ratio_range=[0.5])
        x = torch.ones(2, 4, 500)
        masked_x, mask = aamp(x)
        # 被掩码的点中，80%应该是mask_value(0.0)
        masked_points = masked_x[mask]
        n_zero = (masked_points == 0.0).sum().item()
        n_total = len(masked_points)
        # 大约80%是0（允许一些波动）
        assert n_zero / n_total > 0.6

    def test_mask_ratio_approx(self):
        """测试掩码比例接近设定值"""
        aamp = AAMPMasking(mask_ratio_range=[0.5])
        x = torch.randn(10, 4, 500)
        _, mask = aamp(x)
        actual_ratio = mask.float().mean().item()
        # 实际比例应该接近0.5（允许±0.1的波动）
        assert 0.4 < actual_ratio < 0.6

    def test_return_details(self):
        """测试返回详细信息"""
        aamp = AAMPMasking(mask_ratio_range=[0.3])
        x = torch.randn(2, 4, 500)
        masked_x, mask, details = aamp(x, return_details=True)
        assert "mask_ratio" in details
        assert "n_masked_points" in details
        assert "mask_ratio_actual" in details
        assert details["n_masked_points"] > 0

    def test_deterministic_with_seed(self):
        """测试相同种子产生相同掩码"""
        np.random.seed(42)
        aamp = AAMPMasking(mask_ratio_range=[0.5])
        x = torch.randn(1, 4, 100)
        _, mask1 = aamp(x)

        np.random.seed(42)
        _, mask2 = aamp(x)

        assert torch.equal(mask1, mask2)

    def test_visualize_mask(self):
        """测试可视化数据生成"""
        aamp = AAMPMasking(mask_ratio_range=[0.5])
        x = torch.randn(2, 4, 500)
        masked_x, mask = aamp(x)
        viz_data = AAMPMasking.visualize_mask(x, masked_x, mask, channel=0, batch_idx=0)
        assert "time_axis" in viz_data
        assert "original" in viz_data
        assert "masked" in viz_data
        assert "mask" in viz_data
        assert len(viz_data["time_axis"]) == 500


class TestCompareRandomVsAAMP:
    """测试随机掩码 vs AAMP 对比"""

    def test_shapes(self):
        """测试输出形状"""
        x = torch.randn(2, 4, 500)
        random_masked, random_mask, aamp_masked, aamp_mask = compare_random_vs_aamp(x, mask_ratio=0.5)
        assert random_masked.shape == x.shape
        assert aamp_masked.shape == x.shape
        assert random_mask.shape == x.shape
        assert aamp_mask.shape == x.shape

    def test_both_have_masks(self):
        """测试两种方法都有掩码"""
        x = torch.randn(2, 4, 500)
        _, random_mask, _, aamp_mask = compare_random_vs_aamp(x, mask_ratio=0.5)
        assert random_mask.sum() > 0
        assert aamp_mask.sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
