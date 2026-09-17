"""
双分支注意力模型单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
from attention_model.models.dual_branch_attention import DualBranchAttentionClassifier, ConvBranch


class TestConvBranch:
    """测试单分支卷积网络"""

    def test_init_default(self):
        """测试默认初始化"""
        branch = ConvBranch(in_channels=2, d_model=30)
        assert branch.in_channels == 2
        assert branch.d_model == 30

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        branch = ConvBranch(in_channels=2, d_model=30, conv1_out=32, conv2_out=64)
        x = torch.randn(4, 2, 500)  # [batch, channels, time]
        features, details = branch(x)
        # 多层拼接: conv1_out(32) + conv2_out(64) + conv2_out_std(64) = 160
        assert features.shape == (4, 160)
        assert "embedded" in details
        assert "conv1_out" in details
        assert "conv2_out" in details

    def test_forward_without_multilayer(self):
        """测试不使用多层拼接"""
        branch = ConvBranch(in_channels=2, d_model=30, conv1_out=32, conv2_out=64, use_multilayer_concat=False)
        x = torch.randn(4, 2, 500)
        features, _ = branch(x)
        # 只用 mean+std: 64*2 = 128
        assert features.shape == (4, 128)

    def test_gradient_flow(self):
        """测试梯度流动"""
        branch = ConvBranch(in_channels=2, d_model=30)
        x = torch.randn(4, 2, 500, requires_grad=True)
        features, _ = branch(x)
        loss = features.sum()
        loss.backward()
        assert x.grad is not None


class TestDualBranchAttentionClassifier:
    """测试双分支注意力分类器"""

    def test_init_default(self):
        """测试默认初始化"""
        model = DualBranchAttentionClassifier(n_channels=4, n_classes=3)
        assert model.n_channels == 4
        assert model.left_indices == [0, 1]
        assert model.right_indices == [2, 3]

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        model = DualBranchAttentionClassifier(
            n_channels=4, d_model=30, conv1_out=32, conv2_out=64,
            fusion_hidden=64, n_classes=3,
        )
        x = torch.randn(8, 4, 500)  # [batch, channels, time]
        output = model(x)
        assert "logits" in output
        assert output["logits"].shape == (8, 3)
        assert "left_features" in output
        assert "right_features" in output
        assert "fused_features" in output

    def test_branch_out_dim(self):
        """测试分支输出维度"""
        model = DualBranchAttentionClassifier(
            n_channels=4, d_model=30, conv1_out=32, conv2_out=64,
            use_multilayer_concat=True,
        )
        # 多层拼接: 32 + 64 + 64 = 160
        assert model.branch_out_dim == 160

    def test_fusion_in_dim(self):
        """测试融合层输入维度"""
        model = DualBranchAttentionClassifier(
            n_channels=4, d_model=30, conv1_out=32, conv2_out=64,
            use_diff_feature=True, use_product_feature=True,
        )
        # left(160) + right(160) + diff(160) + product(160) = 640
        assert model.fusion_in_dim == 640

    def test_fusion_without_diff_product(self):
        """测试不使用差值和乘积特征"""
        model = DualBranchAttentionClassifier(
            n_channels=4, d_model=30, conv1_out=32, conv2_out=64,
            use_diff_feature=False, use_product_feature=False,
        )
        # left(160) + right(160) = 320
        assert model.fusion_in_dim == 320

    def test_with_regression_head(self):
        """测试带回归头"""
        model = DualBranchAttentionClassifier(
            n_channels=4, n_classes=3, include_regression_head=True,
        )
        x = torch.randn(4, 4, 500)
        output = model(x)
        assert "regression" in output
        assert output["regression"].shape == (4, 1)

    def test_count_parameters(self):
        """测试参数量统计"""
        model = DualBranchAttentionClassifier(n_channels=4, d_model=30)
        n_params = model.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_get_model_info(self):
        """测试模型信息"""
        model = DualBranchAttentionClassifier(n_channels=4, d_model=30)
        info = model.get_model_info()
        assert info["architecture"] == "dual_branch_attention"
        assert "n_parameters" in info
        assert "use_3d_embedding" in info
        assert "use_iilp_pooling" in info

    def test_gradient_flow(self):
        """测试梯度流动"""
        model = DualBranchAttentionClassifier(n_channels=4, n_classes=3)
        x = torch.randn(4, 4, 500, requires_grad=True)
        output = model(x)
        loss = output["logits"].sum()
        loss.backward()
        assert x.grad is not None
        # 左右分支都应该有梯度
        assert model.left_branch.conv1.weight.grad is not None
        assert model.right_branch.conv1.weight.grad is not None

    def test_branches_not_shared(self):
        """测试左右分支参数不共享"""
        model = DualBranchAttentionClassifier(n_channels=4, d_model=30)
        # 左右分支的 conv1 权重应该不同
        assert not torch.equal(
            model.left_branch.conv1.weight,
            model.right_branch.conv1.weight,
        )

    def test_binary_classification(self):
        """测试二分类"""
        model = DualBranchAttentionClassifier(n_channels=4, n_classes=2)
        x = torch.randn(4, 4, 500)
        output = model(x)
        assert output["logits"].shape == (4, 2)

    def test_extra_repr(self):
        """测试 extra_repr"""
        model = DualBranchAttentionClassifier(n_channels=4, d_model=30)
        repr_str = repr(model)
        assert "n_channels=4" in repr_str
        assert "left=[0, 1]" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
