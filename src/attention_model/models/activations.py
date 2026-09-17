"""
公共激活函数和 FFN 模块

包含：
1. SwiGLU: Swish + Gated Linear Unit，LLaMA/Mistral 标准 FFN 结构
2. get_activation: 激活函数工厂函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数（Swish + Gated Linear Unit）

    LLaMA、Mistral、PaLM 等大模型的标准 FFN 结构，比 GELU 表达能力更强：
        output = w3(SiLU(w1(x)) * w2(x))

    其中 w1 和 w2 是两个独立的线性投影，w1 的输出经过 SiLU(Swish) 后与 w2 逐元素相乘，
    形成门控机制，再由 w3 投影回输出维度。

    相比标准的 Linear→GELU→Linear：
    - 多了一个门控路径（w2），表达能力更强
    - 参数量约多 50%，通常把 hidden_features 缩小到 2/3 来保持总参数量相当
    - 实际效果在大模型中普遍优于 GELU FFN

    Args:
        in_features: 输入维度
        hidden_features: 隐藏维度（默认 in_features * 4/3，补偿门控的额外参数）
        out_features: 输出维度（默认等于 in_features）
        bias: 是否使用偏置
        dropout: dropout 比例（加在 w3 之前）
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or int(in_features * 4 / 3)
        out_features = out_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w1 经过 SiLU 门控，与 w2 逐元素相乘，再由 w3 投影
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


def get_activation(name: str) -> nn.Module:
    """
    激活函数工厂函数

    Args:
        name: 激活函数名称 (silu/swish, gelu, relu, leaky_relu, elu, identity)

    Returns:
        激活函数模块
    """
    name = name.lower()
    if name in ("silu", "swish"):
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"未知的激活函数: {name}，支持: silu, gelu, relu, leaky_relu, elu, identity")
