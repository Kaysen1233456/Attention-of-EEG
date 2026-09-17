"""
3D 电极嵌入模块
借鉴 NeurIPS 2025 NeurIPT 的 3D Electrode Embedding 设计：
- 利用电极的三维物理坐标 (x, y, z) 进行空间编码
- 支持任意通道数和任意电极布局（跨数据集泛化的关键）
- 每个坐标用 sin/cos 位置编码，三个坐标拼接
"""
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class Electrode3DEmbedding(nn.Module):
    """
    3D 电极嵌入层

    将每个 (时间点, 通道) 的 EEG 数据点转换为同时包含：
    1. 信号本身的线性投影
    2. 时间位置编码（sin/cos）
    3. 3D 空间位置编码（基于电极三维坐标的 sin/cos）

    设计参考 NeurIPS 2025 NeurIPT 论文公式 (1)(2)。

    Args:
        d_model: 嵌入维度，必须能被 3 整除（三个坐标各占 d_model/3）
        n_channels: 通道数（用于预分配位置编码，也可以运行时动态计算）
        channel_positions: 每个通道的三维坐标 [(x,y,z), ...]
        max_time_steps: 最大时间步数（用于预分配时间位置编码）
        position_encoding_type: "sin_cos" 或 "learnable"
        dropout: dropout 比例
    """

    def __init__(
        self,
        d_model: int = 32,
        n_channels: int = 4,
        channel_positions: Optional[List[Tuple[float, float, float]]] = None,
        max_time_steps: int = 2000,
        position_encoding_type: str = "sin_cos",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % 3 == 0, f"d_model ({d_model}) 必须能被 3 整除，用于三个坐标的编码"

        self.d_model = d_model
        self.d_per_coord = d_model // 3  # 每个坐标用多少维编码
        self.n_channels = n_channels
        self.max_time_steps = max_time_steps
        self.position_encoding_type = position_encoding_type

        # 默认耳内4通道坐标
        if channel_positions is None:
            channel_positions = [
                (-1.0, 0.5, 0.5),   # L04
                (-1.0, 0.5, -0.5),  # L05
                (1.0, 0.5, 0.5),    # R04
                (1.0, 0.5, -0.5),   # R05
            ]
        if len(channel_positions) != n_channels:
            raise ValueError(
                f"channel_positions length ({len(channel_positions)}) must equal n_channels ({n_channels})"
            )
        self.channel_positions = channel_positions

        # 1. 信号线性投影：把每个时间点的标量信号投影到 d_model 维
        self.signal_projection = nn.Linear(1, d_model)

        # 2. 时间位置编码
        if position_encoding_type == "sin_cos":
            self.time_pe = self._build_sin_cos_pe(max_time_steps, d_model)
            self.register_buffer("time_pe_buffer", self.time_pe, persistent=False)
        elif position_encoding_type == "learnable":
            self.time_pe = nn.Parameter(torch.randn(max_time_steps, d_model) * 0.02)
        else:
            raise ValueError(f"未知的位置编码类型: {position_encoding_type}")

        # 3. 3D 空间位置编码（预计算，因为电极坐标是固定的）
        spatial_pe = self._build_3d_spatial_pe(channel_positions, self.d_per_coord)
        self.register_buffer("spatial_pe_buffer", spatial_pe, persistent=False)  # [n_channels, d_model]

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.signal_projection.weight)
        nn.init.zeros_(self.signal_projection.bias)

    @staticmethod
    def _build_sin_cos_pe(n_pos: int, d_model: int) -> torch.Tensor:
        """
        构建标准 sin/cos 位置编码（Vaswani et al. 2017）

        支持奇数 d_model：sin 用偶数索引（比 cos 多一个），cos 用奇数索引。
        div_term 长度取 sin 和 cos 中较大的，赋值时用切片匹配各自的长度。

        Args:
            n_pos: 位置数量
            d_model: 编码维度

        Returns:
            pe: [n_pos, d_model]
        """
        pe = torch.zeros(n_pos, d_model)
        position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)  # [n_pos, 1]

        # div_term 长度取 sin 索引的数量（偶数索引，比 cos 多一个当 d_model 为奇数时）
        n_sin = (d_model + 1) // 2
        div_term = torch.exp(
            torch.arange(0, n_sin, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # sin 用偶数索引：0, 2, 4, ...
        sin_indices = torch.arange(0, d_model, 2)
        pe[:, sin_indices] = torch.sin(position * div_term[:len(sin_indices)])

        # cos 用奇数索引：1, 3, 5, ...（当 d_model 为奇数时比 sin 少一个）
        cos_indices = torch.arange(1, d_model, 2)
        pe[:, cos_indices] = torch.cos(position * div_term[:len(cos_indices)])

        return pe

    @staticmethod
    def _build_3d_spatial_pe(
        positions: List[Tuple[float, float, float]],
        d_per_coord: int,
    ) -> torch.Tensor:
        """
        构建 3D 空间位置编码

        对每个通道的 (x, y, z) 坐标分别做 sin/cos 编码，然后拼接。
        参考 NeurIPT 论文公式 (1)：PE^(s)_d = Concat(PE_x(x_d), PE_y(y_d), PE_z(z_d))

        Args:
            positions: [(x,y,z), ...]，长度为 n_channels
            d_per_coord: 每个坐标用多少维编码

        Returns:
            spatial_pe: [n_channels, d_per_coord*3]
        """
        n_channels = len(positions)
        spatial_pe = torch.zeros(n_channels, d_per_coord * 3)

        # div_term 的长度必须足够覆盖 sin 和 cos 的位置数
        # sin 用偶数索引 (0,2,4,...)，cos 用奇数索引 (1,3,5,...)
        # 当 d_per_coord 为奇数时，sin 位置比 cos 多一个，所以 div_term 长度取 (d_per_coord + 1) // 2
        n_sin_positions = (d_per_coord + 1) // 2
        div_term = torch.exp(
            torch.arange(0, n_sin_positions, dtype=torch.float) * (-math.log(10000.0) / d_per_coord)
        )

        def _encode_coord(value: float) -> torch.Tensor:
            """对单个坐标值做 sin/cos 编码，返回 [d_per_coord]"""
            pe = torch.zeros(d_per_coord)
            # sin 用偶数索引
            sin_indices = torch.arange(0, d_per_coord, 2)
            pe[sin_indices] = torch.sin(torch.tensor(value) * div_term[:len(sin_indices)])
            # cos 用奇数索引
            cos_indices = torch.arange(1, d_per_coord, 2)
            pe[cos_indices] = torch.cos(torch.tensor(value) * div_term[:len(cos_indices)])
            return pe

        for i, (x, y, z) in enumerate(positions):
            x_pe = _encode_coord(x)
            y_pe = _encode_coord(y)
            z_pe = _encode_coord(z)
            spatial_pe[i] = torch.cat([x_pe, y_pe, z_pe], dim=0)

        return spatial_pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: EEG 信号，形状 [batch, n_channels, time] 或 [batch, time, n_channels]

        Returns:
            embedded: 嵌入后的特征，形状 [batch, n_channels, time, d_model]
                     （保持输入的通道-时间顺序）
        """
        # 统一为 [batch, n_channels, time]
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            # 可能是 [batch, time, n_channels]，转置
            x = x.transpose(1, 2)

        batch, n_channels, time = x.shape

        # 1. 信号线性投影：[batch, n_channels, time, 1] -> [batch, n_channels, time, d_model]
        signal_emb = self.signal_projection(x.unsqueeze(-1))

        # 2. 时间位置编码：[time, d_model] -> [1, 1, time, d_model]
        if self.position_encoding_type == "sin_cos":
            time_pe = self.time_pe_buffer[:time].unsqueeze(0).unsqueeze(0)
        else:
            time_pe = self.time_pe[:time].unsqueeze(0).unsqueeze(0)

        # 3. 空间位置编码：[n_channels, d_model] -> [1, n_channels, 1, d_model]
        if n_channels != self.spatial_pe_buffer.shape[0]:
            raise ValueError(f"input has {n_channels} channels but embedding has {self.spatial_pe_buffer.shape[0]} coordinates")
        if n_channels <= self.spatial_pe_buffer.shape[0]:
            spatial_pe = self.spatial_pe_buffer[:n_channels].unsqueeze(0).unsqueeze(2)
        else:
            # 运行时通道数超过预分配，动态计算
            spatial_pe = self._build_3d_spatial_pe(
                self.channel_positions[:n_channels], self.d_per_coord
            ).to(x.device).unsqueeze(0).unsqueeze(2)

        # 三者相加（参考 NeurIPT 公式 2）
        embedded = signal_emb + time_pe + spatial_pe

        # Dropout
        embedded = self.dropout(embedded)

        return embedded

    def get_spatial_pe(self, n_channels: Optional[int] = None) -> torch.Tensor:
        """获取空间位置编码（用于可视化或调试）"""
        if n_channels is None:
            n_channels = self.n_channels
        return self.spatial_pe_buffer[:n_channels]

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_per_coord={self.d_per_coord}, "
            f"n_channels={self.n_channels}, position_encoding={self.position_encoding_type}"
        )
