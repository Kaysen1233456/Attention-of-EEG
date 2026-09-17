"""
双分支注意力分类模型（阶段一核心模型）

设计思想：
1. 双分支结构：左耳分支和右耳分支分别处理，参数不共享
   - 耳内 EEG 天然有左右对称性，听觉注意和左右侧密切相关
   - 参考 NeurIPS 2025 NeurIPT 的 IILP（脑叶内-脑叶间池化）思想
   - 我们的场景下：左耳内池化（L04+L05）、右耳内池化（R04+R05）、左右间拼接

2. 3D 电极嵌入：每个通道用 (x,y,z) 三维坐标做 sin/cos 位置编码
   - 参考 NeurIPT 的 3D Electrode Embedding
   - 支持跨数据集、跨通道配置的泛化

3. 多层特征拼接：把第1层和第2层的特征拼接后再分类
   - 参考 NeurIPT IILP 的多层 encoder 输出拼接
   - 同时利用低级特征（波形、频率）和高级特征（语义、任务相关）

4. 融合特征：拼接 [左耳特征, 右耳特征, |左耳-右耳|, 左耳*右耳]
   - 差值捕捉左右不对称性
   - 乘积捕捉左右交互
   - 这是双分支网络的标准技巧

参数量：约 40K-60K（NPU 部署友好，和 AAD Gen2.0 的 35682 同量级）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .embedding_3d import Electrode3DEmbedding
from .activations import SwiGLU, get_activation
from .mini_neuript import MiniNeurIPT


class ConvBranch(nn.Module):
    """
    单分支卷积网络（左耳或右耳）

    结构参考 AAD Gen2.0 的 mean_std_pool 架构：
    Linear(通道→d_model) → Conv1d(k=5,s=2) → GELU → Conv1d(k=3,s=2) → GELU → 池化

    Args:
        in_channels: 输入通道数（左耳2通道或右耳2通道）
        d_model: 嵌入维度
        conv1_out: 第1层卷积输出通道
        conv1_kernel: 第1层卷积核大小
        conv1_stride: 第1层卷积步长
        conv2_out: 第2层卷积输出通道
        conv2_kernel: 第2层卷积核大小
        conv2_stride: 第2层卷积步长
        activation: 激活函数类型 (silu=Swish, gelu, relu)。默认silu，SwiGLU的门控基础
        dropout: dropout 比例
        use_3d_embedding: 是否使用3D电极嵌入
        channel_positions: 该分支的通道三维坐标
        use_multilayer_concat: 是否使用多层特征拼接
        channel_reduce: 3D嵌入后通道降维方式
            - "none": 不平均，各通道各自过卷积（保留3D电极身份信息，推荐）
            - "mean": 对通道维取平均（简单但抹掉通道差异）
            - "flatten_linear": flatten后Linear投影回d_model
    """

    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 33,  # 必须能被3整除（3D坐标各占1/3），默认33
        conv1_out: int = 32,
        conv1_kernel: int = 5,
        conv1_stride: int = 2,
        conv2_out: int = 64,
        conv2_kernel: int = 3,
        conv2_stride: int = 2,
        activation: str = "silu",
        dropout: float = 0.0,
        use_3d_embedding: bool = True,
        channel_positions: Optional[List[Tuple[float, float, float]]] = None,
        use_multilayer_concat: bool = True,
        channel_reduce: str = "none",  # none / mean / flatten_linear
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.use_3d_embedding = use_3d_embedding
        self.use_multilayer_concat = use_multilayer_concat
        self.channel_reduce = channel_reduce

        assert channel_reduce in ("none", "mean", "flatten_linear"), \
            f"channel_reduce 必须是 'none'/'mean'/'flatten_linear'，当前为 {channel_reduce}"

        # 激活函数（默认 silu=Swish，SwiGLU 的门控基础）
        self.act = get_activation(activation)

        # 3D 电极嵌入（可选）
        if use_3d_embedding:
            if channel_positions is None:
                default_positions = [
                    (-1.0, 0.5, 0.5), (-1.0, 0.5, -0.5),
                    (1.0, 0.5, 0.5), (1.0, 0.5, -0.5),
                ]
                channel_positions = default_positions[:in_channels]
            self.embedding = Electrode3DEmbedding(
                d_model=d_model,
                n_channels=in_channels,
                channel_positions=channel_positions,
                position_encoding_type="sin_cos",
                dropout=dropout,
            )
            # 嵌入后形状 [batch, channels, time, d_model]
            # 根据 channel_reduce 选择不同的降维方式
            if channel_reduce == "flatten_linear":
                # flatten 后 Linear 投影回 d_model
                self.channel_flatten_proj = nn.Linear(in_channels * d_model, d_model)
            # none: 不做额外处理，forward 中 reshape 成 [batch, channels*d_model, time]
            # mean: forward 中对通道维取平均
        else:
            # 不用3D嵌入时，用简单的1x1卷积把通道投影到d_model
            self.channel_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

        # 第1层卷积的输入通道数
        if use_3d_embedding and channel_reduce == "none":
            conv1_in_channels = in_channels * d_model
        else:
            conv1_in_channels = d_model

        # 第1层卷积
        self.conv1 = nn.Conv1d(
            conv1_in_channels, conv1_out,
            kernel_size=conv1_kernel,
            stride=conv1_stride,
            padding=conv1_kernel // 2,
        )
        # BatchNorm（默认不使用，AAD实验证明在被试级任务上有害；
        # 注意力任务是窗口级分类，如需启用可设 use_bn=True）
        self.use_bn = False
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(conv1_out)

        # 第2层卷积
        self.conv2 = nn.Conv1d(
            conv1_out, conv2_out,
            kernel_size=conv2_kernel,
            stride=conv2_stride,
            padding=conv2_kernel // 2,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 输出维度
        if use_multilayer_concat:
            # 多层拼接：第1层GAP(conv1_out) + 第2层GAP(conv2_out) + 第2层std(conv2_out)
            self.out_dim = conv1_out + conv2_out * 2
        else:
            # 只用第2层的 mean_std_pool
            self.out_dim = conv2_out * 2  # mean + std

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            x: 该分支的EEG信号，形状 [batch, in_channels, time]

        Returns:
            features: 池化后的特征向量 [batch, out_dim]
            details: 中间特征字典（用于调试/可视化）
        """
        details = {}
        batch, channels, time = x.shape

        # 1. 嵌入层
        if self.use_3d_embedding:
            # [batch, channels, time] -> [batch, channels, time, d_model]
            embedded = self.embedding(x)

            if self.channel_reduce == "mean":
                # 对通道维取平均，得到 [batch, time, d_model] -> [batch, d_model, time]
                # （简单但抹掉通道间差异）
                embedded = embedded.mean(dim=1)  # [batch, time, d_model]
                x = embedded.transpose(1, 2)  # [batch, d_model, time]
            elif self.channel_reduce == "flatten_linear":
                # flatten 通道和d_model，再 Linear 投影回 d_model
                # [batch, channels, time, d_model] -> [batch, time, channels*d_model]
                batch, ch, time, dm = embedded.shape
                embedded = embedded.permute(0, 2, 1, 3).reshape(batch, time, ch * dm)
                embedded = self.channel_flatten_proj(embedded)  # [batch, time, d_model]
                x = embedded.transpose(1, 2)  # [batch, d_model, time]
            else:  # channel_reduce == "none"
                # 不平均，各通道各自过卷积（保留3D电极身份信息）
                # [batch, channels, time, d_model] -> [batch, channels*d_model, time]
                batch, ch, time, dm = embedded.shape
                x = embedded.permute(0, 1, 3, 2).reshape(batch, ch * dm, time)
        else:
            x = self.channel_proj(x)  # [batch, d_model, time]

        details["embedded"] = x.detach()

        # 2. 第1层卷积
        x1 = self.conv1(x)
        if self.use_bn:
            x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.dropout(x1)
        details["conv1_out"] = x1.detach()

        # 3. 第2层卷积
        x2 = self.conv2(x1)
        x2 = self.act(x2)
        x2 = self.dropout(x2)
        details["conv2_out"] = x2.detach()

        # 4. 池化
        if self.use_multilayer_concat:
            # 第1层：全局平均池化
            feat1 = F.adaptive_avg_pool1d(x1, 1).squeeze(-1)  # [batch, conv1_out]

            # 第2层：全局平均池化
            feat2_mean = F.adaptive_avg_pool1d(x2, 1).squeeze(-1)  # [batch, conv2_out]

            # 第2层：全局标准差池化（mean_std_pool 的思想）
            feat2_std = torch.sqrt(x2.var(dim=-1) + 1e-8)  # [batch, conv2_out]

            # 拼接
            features = torch.cat([feat1, feat2_mean, feat2_std], dim=-1)
        else:
            # 只用第2层的 mean + std
            feat_mean = F.adaptive_avg_pool1d(x2, 1).squeeze(-1)
            feat_std = torch.sqrt(x2.var(dim=-1) + 1e-8)
            features = torch.cat([feat_mean, feat_std], dim=-1)

        details["features"] = features.detach()
        return features, details


class DualBranchAttentionClassifier(nn.Module):
    """
    双分支注意力分类器（阶段一主模型）

    架构：
    输入 EEG [batch, 4, 500]
        │
        ├── 左耳分支: 取通道 0,1 (L04, L05) → ConvBranch → 左耳特征
        ├── 右耳分支: 取通道 2,3 (R04, R05) → ConvBranch → 右耳特征
        │
        └── 融合:
            拼接 [左耳特征, 右耳特征, |左耳-右耳|, 左耳*右耳]
                │
                ▼
            MLP: fusion_in → fusion_hidden → n_classes (分类)
            并行分支: fusion_in → 64 → 1 (认知负荷回归，可选)

    设计参考：
    - NeurIPS 2025 NeurIPT 的 IILP（脑叶内-脑叶间池化）
    - AAD Gen2.0 的 mean_std_pool 架构
    - 双分支网络的标准融合技巧（差值+乘积）

    Args:
        n_channels: 总通道数（默认4）
        left_indices: 左耳通道索引
        right_indices: 右耳通道索引
        d_model: 嵌入维度
        conv1_out: 第1层卷积输出通道
        conv2_out: 第2层卷积输出通道
        fusion_hidden: 融合层隐藏维度
        n_classes: 分类类别数
        activation: 激活函数
        dropout: dropout 比例
        use_3d_embedding: 是否使用3D电极嵌入
        channel_reduce: 3D嵌入后通道降维方式 (none/mean/flatten_linear)
        use_multilayer_concat: 是否使用多层特征拼接
        use_diff_feature: 是否使用 |左耳-右耳| 差值特征
        use_product_feature: 是否使用 左耳*右耳 乘积特征
        channel_positions: 所有通道的三维坐标
        include_regression_head: 是否包含认知负荷回归头
    """

    def __init__(
        self,
        n_channels: int = 4,
        left_indices: Optional[List[int]] = None,
        right_indices: Optional[List[int]] = None,
        d_model: int = 33,  # 必须能被3整除（3D坐标各占1/3），默认33
        conv1_out: int = 32,
        conv2_out: int = 64,
        fusion_hidden: int = 64,
        n_classes: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        use_3d_embedding: bool = True,
        channel_reduce: str = "none",  # none / mean / flatten_linear
        use_multilayer_concat: bool = True,
        use_diff_feature: bool = True,
        use_product_feature: bool = True,
        channel_positions: Optional[List[Tuple[float, float, float]]] = None,
        include_regression_head: bool = False,
        use_iilp_pooling: bool = True,
        pretrained_path: Optional[str] = None,
        pretrained_n_heads: int = 8,
        pretrained_n_layers: int = 4,
        pretrained_d_ff: int = 384,
        freeze_pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.left_indices = left_indices if left_indices is not None else [0, 1]
        self.right_indices = right_indices if right_indices is not None else [2, 3]
        self.use_3d_embedding = use_3d_embedding
        self.channel_reduce = channel_reduce
        self.use_diff_feature = use_diff_feature
        self.use_product_feature = use_product_feature
        self.include_regression_head = include_regression_head
        self.use_iilp_pooling = use_iilp_pooling
        self.pretrained_path = pretrained_path
        self.use_pretrained_backbone = pretrained_path is not None

        # d_model 必须能被3整除（3D坐标各占1/3），自动调整到最近的合法值
        if use_3d_embedding and d_model % 3 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 3 when using 3D embedding")
        self.d_model = d_model

        # 默认耳内4通道坐标
        if channel_positions is None:
            channel_positions = [
                (-1.0, 0.5, 0.5),   # L04
                (-1.0, 0.5, -0.5),  # L05
                (1.0, 0.5, 0.5),    # R04
                (1.0, 0.5, -0.5),   # R05
            ]
        self.channel_positions = channel_positions

        self.pretrained_backbone = None
        if self.use_pretrained_backbone:
            self.pretrained_backbone = MiniNeurIPT(
                d_model=d_model,
                n_heads=pretrained_n_heads,
                d_ff=pretrained_d_ff,
                n_layers=pretrained_n_layers,
                n_channels=n_channels,
                channel_positions=channel_positions,
            )
            state = torch.load(pretrained_path, map_location="cpu")
            self.pretrained_backbone.load_state_dict(state, strict=True)
            if freeze_pretrained_backbone:
                for parameter in self.pretrained_backbone.parameters():
                    parameter.requires_grad = False

        # 左耳分支的通道坐标
        left_positions = [channel_positions[i] for i in self.left_indices]
        right_positions = [channel_positions[i] for i in self.right_indices]

        # 左耳分支（参数独立）
        self.left_branch = ConvBranch(
            in_channels=len(self.left_indices),
            d_model=d_model,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            activation=activation,
            dropout=dropout,
            use_3d_embedding=use_3d_embedding,
            channel_positions=left_positions,
            use_multilayer_concat=use_multilayer_concat,
            channel_reduce=channel_reduce,
        )

        # 右耳分支（参数独立，不共享）
        self.right_branch = ConvBranch(
            in_channels=len(self.right_indices),
            d_model=d_model,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            activation=activation,
            dropout=dropout,
            use_3d_embedding=use_3d_embedding,
            channel_positions=right_positions,
            use_multilayer_concat=use_multilayer_concat,
            channel_reduce=channel_reduce,
        )

        # 计算融合层输入维度
        branch_out_dim = d_model if self.use_pretrained_backbone else self.left_branch.out_dim
        fusion_in_dim = branch_out_dim * 2  # 左耳 + 右耳
        if use_diff_feature:
            fusion_in_dim += branch_out_dim  # |左耳-右耳|
        if use_product_feature:
            fusion_in_dim += branch_out_dim  # 左耳*右耳

        self.branch_out_dim = branch_out_dim
        self.fusion_in_dim = fusion_in_dim

        # 融合 MLP（分类头）— 使用 SwiGLU 结构
        # SwiGLU 的 hidden_features 通常取 in_features * 4/3 补偿门控的额外参数
        swiglu_hidden = int(fusion_hidden * 4 / 3)
        self.fusion = nn.Sequential(
            SwiGLU(fusion_in_dim, hidden_features=swiglu_hidden, out_features=fusion_hidden),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden, n_classes),
        )

        # 认知负荷回归头（可选）— 同样使用 SwiGLU
        if include_regression_head:
            self.regression_head = nn.Sequential(
                SwiGLU(fusion_in_dim, hidden_features=int(64 * 4 / 3), out_features=64),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(64, 1),
            )

        self._init_weights()

    def _init_weights(self):
        """初始化融合层权重"""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if self.include_regression_head:
            for m in self.regression_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: EEG 信号，形状 [batch, n_channels, time]

        Returns:
            输出字典：
            - logits: 分类logits [batch, n_classes]
            - left_features: 左耳分支特征 [batch, branch_out_dim]
            - right_features: 右耳分支特征 [batch, branch_out_dim]
            - fused_features: 融合后特征 [batch, fusion_in_dim]
            - regression (可选): 认知负荷预测 [batch, 1]
        """
        # 1. 拆分左右耳
        if self.use_pretrained_backbone:
            encoded = self.pretrained_backbone.encode(x)
            left_features = encoded[:, self.left_indices].mean(dim=(1, 2))
            right_features = encoded[:, self.right_indices].mean(dim=(1, 2))
            fusion_parts = [left_features, right_features]
            if self.use_diff_feature:
                fusion_parts.append(torch.abs(left_features - right_features))
            if self.use_product_feature:
                fusion_parts.append(left_features * right_features)
            fused = torch.cat(fusion_parts, dim=-1)
            logits = self.fusion(fused)
            return {"logits": logits, "left_features": left_features,
                    "right_features": right_features, "fused_features": fused}

        left_x = x[:, self.left_indices, :]   # [batch, 2, time]
        right_x = x[:, self.right_indices, :]  # [batch, 2, time]

        # 2. 左右分支分别处理（IILP 思想：脑叶内独立处理）
        left_features, left_details = self.left_branch(left_x)
        right_features, right_details = self.right_branch(right_x)

        # 3. 融合（IILP 思想：脑叶间拼接）
        fusion_parts = [left_features, right_features]

        if self.use_diff_feature:
            # 差值特征：捕捉左右不对称性
            diff = torch.abs(left_features - right_features)
            fusion_parts.append(diff)

        if self.use_product_feature:
            # 乘积特征：捕捉左右交互
            product = left_features * right_features
            fusion_parts.append(product)

        fused = torch.cat(fusion_parts, dim=-1)  # [batch, fusion_in_dim]

        # 4. 分类头
        logits = self.fusion(fused)  # [batch, n_classes]

        output = {
            "logits": logits,
            "left_features": left_features,
            "right_features": right_features,
            "fused_features": fused,
        }

        # 5. 回归头（可选）
        if self.include_regression_head:
            regression = self.regression_head(fused)  # [batch, 1]
            output["regression"] = regression

        return output

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        """获取模型信息（用于日志和配置保存）"""
        return {
            "architecture": "dual_branch_attention",
            "n_channels": self.n_channels,
            "left_indices": self.left_indices,
            "right_indices": self.right_indices,
            "branch_out_dim": self.branch_out_dim,
            "fusion_in_dim": self.fusion_in_dim,
            "n_parameters": self.count_parameters(),
            "use_3d_embedding": self.use_3d_embedding,
            "channel_reduce": self.channel_reduce,
            "use_iilp_pooling": self.use_iilp_pooling,
            "use_diff_feature": self.use_diff_feature,
            "use_product_feature": self.use_product_feature,
            "use_iilp_pooling": self.use_iilp_pooling,
            "use_pretrained_backbone": self.use_pretrained_backbone,
            "pretrained_path": self.pretrained_path,
        }

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, left={self.left_indices}, right={self.right_indices}, "
            f"branch_out={self.branch_out_dim}, fusion_in={self.fusion_in_dim}, "
            f"params={self.count_parameters()}"
        )


class SingleBranchAttentionClassifier(nn.Module):
    """
    单分支注意力分类器（用于消融实验，对比双分支的价值）

    架构：
    输入 EEG [batch, n_channels, 500]
        │
        ▼
    ConvBranch（所有通道一起处理，不分左右）
        │
        ▼
    SwiGLU 分类头 → logits [batch, n_classes]

    设计目的：
    - 与 DualBranchAttentionClassifier 做公平对比，验证 IILP（双分支）的价值
    - 唯一区别：单分支所有通道一起过卷积，双分支左右分开过卷积再融合
    - 主干结构（ConvBranch）完全一致，保证对比公平

    Args:
        n_channels: 总通道数（默认4）
        d_model: 嵌入维度
        conv1_out: 第1层卷积输出通道
        conv2_out: 第2层卷积输出通道
        hidden_dim: 分类头隐藏维度
        n_classes: 分类类别数
        activation: 激活函数
        dropout: dropout 比例
        use_3d_embedding: 是否使用3D电极嵌入
        channel_reduce: 3D嵌入后通道降维方式 (none/mean/flatten_linear)
        use_multilayer_concat: 是否使用多层特征拼接
        channel_positions: 所有通道的三维坐标
    """

    def __init__(
        self,
        n_channels: int = 4,
        d_model: int = 33,
        conv1_out: int = 32,
        conv2_out: int = 64,
        hidden_dim: int = 64,
        n_classes: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        use_3d_embedding: bool = True,
        channel_reduce: str = "none",
        use_multilayer_concat: bool = True,
        channel_positions: Optional[List[Tuple[float, float, float]]] = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.use_3d_embedding = use_3d_embedding
        self.channel_reduce = channel_reduce

        # d_model 必须能被3整除，自动调整
        if use_3d_embedding and d_model % 3 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 3 when using 3D embedding")
        self.d_model = d_model

        # 默认耳内4通道坐标
        if channel_positions is None:
            channel_positions = [
                (-1.0, 0.5, 0.5),   # L04
                (-1.0, 0.5, -0.5),  # L05
                (1.0, 0.5, 0.5),    # R04
                (1.0, 0.5, -0.5),   # R05
            ]
        self.channel_positions = channel_positions

        # 单分支主干（复用 ConvBranch，处理所有通道）
        self.backbone = ConvBranch(
            in_channels=n_channels,
            d_model=d_model,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            activation=activation,
            dropout=dropout,
            use_3d_embedding=use_3d_embedding,
            channel_positions=channel_positions,
            use_multilayer_concat=use_multilayer_concat,
            channel_reduce=channel_reduce,
        )

        self.backbone_out_dim = self.backbone.out_dim

        # 分类头（SwiGLU 结构）
        swiglu_hidden = int(hidden_dim * 4 / 3)
        self.classifier = nn.Sequential(
            SwiGLU(self.backbone_out_dim, hidden_features=swiglu_hidden, out_features=hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: EEG 信号，形状 [batch, n_channels, time]

        Returns:
            输出字典：
            - logits: 分类logits [batch, n_classes]
            - features: 主干特征 [batch, backbone_out_dim]
        """
        features, details = self.backbone(x)
        logits = self.classifier(features)

        return {
            "logits": logits,
            "features": features,
        }

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "architecture": "single_branch_attention",
            "n_channels": self.n_channels,
            "backbone_out_dim": self.backbone_out_dim,
            "n_parameters": self.count_parameters(),
            "use_3d_embedding": self.use_3d_embedding,
            "channel_reduce": self.channel_reduce,
        }

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, backbone_out={self.backbone_out_dim}, "
            f"params={self.count_parameters()}"
        )
