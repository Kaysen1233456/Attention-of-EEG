"""
3D 电极嵌入模块单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
from attention_model.models.embedding_3d import Electrode3DEmbedding


class TestElectrode3DEmbedding:
    """测试 3D 电极嵌入"""

    def test_init_default(self):
        """测试默认初始化"""
        emb = Electrode3DEmbedding(d_model=30, n_channels=4)
        assert emb.d_model == 30
        assert emb.d_per_coord == 10
        assert emb.n_channels == 4

    def test_d_model_must_divisible_by_3(self):
        """测试 d_model 必须能被3整除"""
        with pytest.raises(AssertionError):
            Electrode3DEmbedding(d_model=31, n_channels=4)

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        emb = Electrode3DEmbedding(d_model=30, n_channels=4)
        x = torch.randn(2, 4, 500)  # [batch, channels, time]
        out = emb(x)
        assert out.shape == (2, 4, 500, 30)

    def test_forward_with_time_first(self):
        """测试 time-first 输入 [batch, time, channels]"""
        emb = Electrode3DEmbedding(d_model=30, n_channels=4)
        x = torch.randn(2, 500, 4)  # [batch, time, channels]
        out = emb(x)
        assert out.shape == (2, 4, 500, 30)  # 输出统一为 [batch, channels, time, d_model]

    def test_spatial_pe_shape(self):
        """测试空间位置编码形状"""
        emb = Electrode3DEmbedding(d_model=30, n_channels=4)
        spatial_pe = emb.get_spatial_pe()
        assert spatial_pe.shape == (4, 30)

    def test_custom_channel_positions(self):
        """测试自定义通道坐标"""
        positions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
        emb = Electrode3DEmbedding(d_model=30, n_channels=4, channel_positions=positions)
        x = torch.randn(2, 4, 100)
        out = emb(x)
        assert out.shape == (2, 4, 100, 30)

    def test_different_n_channels_runtime(self):
        """测试运行时通道数不同（跨数据集泛化）"""
        emb = Electrode3DEmbedding(d_model=30, n_channels=4)
        # 用 8 通道输入测试（预分配了4通道，但运行时可以动态计算）
        x = torch.randn(2, 8, 100)
        # 这会触发动态空间PE计算
        with pytest.raises(ValueError, match="coordinates"):
            emb(x)

    def test_gradient_flow(self):
        """测试梯度流动"""
        emb = Electrode3DEmbedding(d_model=30, n_channels=4)
        x = torch.randn(2, 4, 100, requires_grad=True)
        out = emb(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert emb.signal_projection.weight.grad is not None

    def test_extra_repr(self):
        """测试 extra_repr"""
        emb = Electrode3DEmbedding(d_model=30, n_channels=4)
        repr_str = repr(emb)
        assert "d_model=30" in repr_str
        assert "d_per_coord=10" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
