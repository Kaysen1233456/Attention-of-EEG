"""
迷你版 NeurIPT 预训练模型（阶段二自监督预训练用）

这是 NeurIPS 2025 NeurIPT 的简化实现，保留核心设计思想：
1. 3D 电极嵌入（跨通道泛化）
2. AAMP 振幅感知掩码（学真正的脑电特征）
3. 分层注意力（时间注意力 + 空间注意力，参考 Crossformer）
4. 渐进式混合专家 PMoE（简化版，深层专家多）
5. 解码器重建被掩码的信号

和原版 NeurIPT 的区别：
- 原版 73.5M 参数，8张RTX4090训练30小时
- 迷你版 ~1-2M 参数，单张A10训练10-20小时
- 简化了PMoE（不做完整的路由，用固定的专家分配）
- 简化了分层注意力（不用完整的Crossformer，用标准的时间+空间注意力）

预训练后可以：
- 微调到下游注意力分类任务
- 知识蒸馏到双分支小模型（<100K参数）用于部署
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .embedding_3d import Electrode3DEmbedding
from .activations import SwiGLU, get_activation
from ..data.aamp_masking import AAMPMasking


class MultiHeadAttention(nn.Module):
    """标准多头注意力"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class SimplifiedMoE(nn.Module):
    """
    简化版渐进式混合专家（PMoE）

    原版 NeurIPT 的 PMoE 用 TopK 路由，这里简化为：
    - 多个专家 FFN 并行计算
    - 用可学习的门控权重加权求和
    - 加上共享专家保证泛化
    - 专家数量随层数增加（渐进式）

    这样简化后不需要复杂的路由计算，但保留了"多专家分工"的核心思想。
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.top_k = max(1, math.ceil(0.5 * n_experts)) if n_experts > 0 else 0
        self.last_load_balancing_loss = torch.tensor(0.0)

        # 共享专家（始终参与，保证泛化）— 使用 SwiGLU 结构
        self.shared_expert = SwiGLU(
            in_features=d_model,
            hidden_features=d_ff,
            out_features=d_model,
            dropout=dropout,
        )

        # 专用专家（n_experts个，门控加权）— 同样使用 SwiGLU
        if n_experts > 0:
            self.experts = nn.ModuleList([
                SwiGLU(
                    in_features=d_model,
                    hidden_features=d_ff,
                    out_features=d_model,
                    dropout=dropout,
                )
                for _ in range(n_experts)
            ])
            # 门控网络
            self.gate = nn.Linear(d_model, n_experts)
        else:
            self.experts = None
            self.gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 共享专家
        shared_out = self.shared_expert(x)

        if self.experts is None or self.n_experts == 0:
            return shared_out

        # 专用专家并行计算
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # [batch, seq, d_model, n_experts]

        # 门控加权
        gate_weights = F.softmax(self.gate(x), dim=-1)
        top_values, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(gate_weights).scatter(-1, top_indices, top_values)
        sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        reduce_dims = tuple(range(sparse_weights.dim() - 1))
        importance = gate_weights.mean(dim=reduce_dims)
        load = (sparse_weights > 0).float().mean(dim=reduce_dims)
        self.last_load_balancing_loss = self.n_experts * torch.sum(importance * load)
        gate_weights = sparse_weights.unsqueeze(-2)  # [batch, seq, 1, n_experts]
        expert_out = (expert_outs * gate_weights).sum(dim=-1)  # [batch, seq, d_model]

        return shared_out + expert_out


class MiniNeurIPTEncoderLayer(nn.Module):
    """
    迷你版 NeurIPT Encoder Layer

    结构（参考 NeurIPT 的分层注意力）：
    1. 时间注意力（对时间维做自注意力）
    2. 残差 + LayerNorm
    3. 空间注意力（对通道维做自注意力）
    4. 残差 + LayerNorm
    5. 简化版 MoE FFN
    6. 残差 + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int = 0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        # 时间注意力
        self.temporal_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.temporal_norm = nn.LayerNorm(d_model)

        # 空间注意力
        self.spatial_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.spatial_norm = nn.LayerNorm(d_model)

        # MoE FFN
        self.moe = SimplifiedMoE(d_model, n_experts, d_ff, dropout, activation)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_channels, time, d_model]
        Returns:
            [batch, n_channels, time, d_model]
        """
        batch, n_channels, time, d_model = x.shape

        # 1. 时间注意力：把通道维当batch，对时间维做注意力
        x_temporal = x.reshape(batch * n_channels, time, d_model)
        x_temporal = self.temporal_attn(x_temporal)
        x_temporal = x_temporal.reshape(batch, n_channels, time, d_model)
        x = self.temporal_norm(x + self.dropout(x_temporal))

        # 2. 空间注意力：把时间维当batch，对通道维做注意力
        x_spatial = x.permute(0, 2, 1, 3).reshape(batch * time, n_channels, d_model)
        x_spatial = self.spatial_attn(x_spatial)
        x_spatial = x_spatial.reshape(batch, time, n_channels, d_model).permute(0, 2, 1, 3)
        x = self.spatial_norm(x + self.dropout(x_spatial))

        # 3. MoE FFN
        x_ffn = x.reshape(batch * n_channels * time, d_model)
        x_ffn = self.moe(x_ffn)
        x_ffn = x_ffn.reshape(batch, n_channels, time, d_model)
        x = self.ffn_norm(x + self.dropout(x_ffn))

        return x


class MiniNeurIPT(nn.Module):
    """
    迷你版 NeurIPT（自监督预训练模型）

    架构：
    输入 EEG [batch, n_channels, time]
        │
        ▼
    3D 电极嵌入 → [batch, n_channels, time, d_model]
        │
        ▼
    AAMP 振幅感知掩码（训练时）
        │
        ▼
    L 层 Encoder（时间注意力 + 空间注意力 + 简化MoE）
        │
        ▼
    解码器：线性层重建被掩码的信号 → [batch, n_channels, time]
        │
        ▼
    损失：L1 重建损失（只在被掩码的点上计算）

    渐进式专家配置（参考 NeurIPT 的 [0,2,2,4,4,6]）：
    第1层: 0专家（只有共享专家）
    第2层: 2专家
    第3层: 2专家
    第4层: 4专家

    Args:
        d_model: 嵌入维度（默认128，比双分支模型大）
        n_heads: 注意力头数
        d_ff: FFN隐藏维度
        n_layers: encoder层数
        expert_config: 每层的专家数列表
        n_channels: 通道数
        channel_positions: 通道三维坐标
        max_time_steps: 最大时间步数
        dropout: dropout
        activation: 激活函数
        mask_ratio_range: AAMP掩码比例范围
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        n_layers: int = 4,
        expert_config: Optional[List[int]] = None,
        n_channels: int = 4,
        channel_positions: Optional[List[Tuple[float, float, float]]] = None,
        max_time_steps: int = 2000,
        dropout: float = 0.1,
        activation: str = "gelu",
        mask_ratio_range: Optional[List[float]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.temporal_pool = 4

        # 默认渐进式专家配置（4层版）
        if d_model % 3 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 3 when using 3D embedding")
        if expert_config is None:
            schedule = [0, 2, 2, 4, 4, 6]
            expert_config = schedule[:n_layers]
            if len(expert_config) < n_layers:
                expert_config.extend([expert_config[-1]] * (n_layers - len(expert_config)))
        if len(expert_config) != n_layers:
            raise ValueError("expert_config length must equal n_layers")

        # 1. 3D 电极嵌入
        self.embedding = Electrode3DEmbedding(
            d_model=d_model,
            n_channels=n_channels,
            channel_positions=channel_positions,
            max_time_steps=max_time_steps,
            position_encoding_type="sin_cos",
            dropout=dropout,
        )

        # 2. AAMP 掩码
        if mask_ratio_range is None:
            mask_ratio_range = [0.2, 0.35, 0.5]
        self.aamp = AAMPMasking(mask_ratio_range=mask_ratio_range)

        # 3. Encoder 层
        self.layers = nn.ModuleList([
            MiniNeurIPTEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_experts=expert_config[i],
                dropout=dropout,
                activation=activation,
            )
            for i in range(n_layers)
        ])

        # 4. 解码器（重建被掩码的信号）— 使用 SiLU 激活
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        # 5. 最终 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        self.last_load_balancing_loss = torch.tensor(0.0)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        apply_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（预训练模式）

        Args:
            x: EEG 信号 [batch, n_channels, time]
            apply_mask: 是否应用 AAMP 掩码（预训练时True，微调/特征提取时False）

        Returns:
            输出字典：
            - reconstruction: 重建的信号 [batch, n_channels, time]
            - mask: 掩码标记 [batch, n_channels, time]
            - encoded: 编码后的特征 [batch, n_channels, time, d_model]
            - loss: 重建损失（只在被掩码的点上）
        """
        original_x = x.clone()

        # 1. AAMP 掩码
        if apply_mask:
            masked_x, mask = self.aamp(x)
        else:
            masked_x = x
            mask = torch.zeros_like(x, dtype=torch.bool)

        encoded = self.encode(masked_x)

        # 4. 解码器重建
        batch, n_channels, encoded_time, d_model = encoded.shape
        decoded = self.decoder(encoded.reshape(-1, d_model))
        reconstruction = decoded.reshape(batch, n_channels, encoded_time)
        if encoded_time != x.shape[-1]:
            reconstruction = F.interpolate(
                reconstruction.reshape(batch * n_channels, 1, encoded_time),
                size=x.shape[-1], mode="linear", align_corners=False,
            ).reshape(batch, n_channels, x.shape[-1])

        # 5. 计算损失（只在被掩码的点上）
        if mask.sum() > 0:
            loss = F.l1_loss(
                reconstruction[mask],
                original_x[mask],
            )
        else:
            loss = torch.tensor(0.0, device=x.device)
        loss = loss + 0.01 * self.last_load_balancing_loss.to(x.device)

        return {
            "reconstruction": reconstruction,
            "mask": mask,
            "encoded": encoded,
            "loss": loss,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an unmasked EEG batch for downstream transfer learning."""
        hidden = self.embedding(x)
        batch, channels, time, dim = hidden.shape
        if self.temporal_pool > 1 and time >= self.temporal_pool:
            hidden = hidden.reshape(batch * channels, time, dim).transpose(1, 2)
            hidden = F.avg_pool1d(hidden, kernel_size=self.temporal_pool, stride=self.temporal_pool)
            hidden = hidden.transpose(1, 2).reshape(batch, channels, -1, dim)
        for layer in self.layers:
            hidden = layer(hidden)
        aux_losses = [
            layer.moe.last_load_balancing_loss
            for layer in self.layers
            if layer.moe.n_experts > 0
        ]
        self.last_load_balancing_loss = (
            torch.stack([value.to(hidden.device) for value in aux_losses]).mean()
            if aux_losses else torch.tensor(0.0, device=hidden.device)
        )
        return self.final_norm(hidden)

    def extract_features(
        self,
        x: torch.Tensor,
        pooling: str = "mean",  # mean / max / mean_std / iilp
        channel_groups: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """
        提取特征（用于微调或蒸馏）

        Args:
            x: EEG 信号 [batch, n_channels, time]
            pooling: 池化方式
                - mean: 全局平均池化
                - max: 全局最大池化
                - mean_std: 均值+标准差
                - iilp: 左右耳分组池化（需要 channel_groups）
            channel_groups: IILP 的通道分组，如 [[0,1],[2,3]]

        Returns:
            features: [batch, feature_dim]
        """
        with torch.no_grad():
            output = self.forward(x, apply_mask=False)
            encoded = output["encoded"]  # [batch, n_channels, time, d_model]

        batch, n_channels, time, d_model = encoded.shape

        if pooling == "mean":
            # 全局平均池化
            features = encoded.mean(dim=(1, 2))  # [batch, d_model]
        elif pooling == "max":
            features = encoded.max(dim=2)[0].max(dim=1)[0]  # [batch, d_model]
        elif pooling == "mean_std":
            feat_mean = encoded.mean(dim=(1, 2))
            feat_std = encoded.reshape(batch, -1, d_model).std(dim=1)
            features = torch.cat([feat_mean, feat_std], dim=-1)  # [batch, d_model*2]
        elif pooling == "iilp":
            # IILP 左右耳分组池化
            if channel_groups is None:
                channel_groups = [[0, 1], [2, 3]]  # 默认左右耳
            group_features = []
            for group in channel_groups:
                group_encoded = encoded[:, group, :, :]  # [batch, group_size, time, d_model]
                group_mean = group_encoded.mean(dim=(1, 2))  # [batch, d_model]
                group_features.append(group_mean)
            features = torch.cat(group_features, dim=-1)  # [batch, d_model*n_groups]
        else:
            raise ValueError(f"未知的池化方式: {pooling}")

        return features

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "architecture": "mini_neuript",
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_parameters": self.count_parameters(),
        }
