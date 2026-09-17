"""
损失函数模块

包含：
1. AttentionLoss: 注意力分类损失（交叉熵 + 可选标签平滑 + 可选一致性正则化）
2. ConsistencyLoss: 一致性正则化损失（同一被试不同窗口的预测应该一致）
3. AAMPReconstructionLoss: AAMP 预训练的重建损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class AttentionLoss(nn.Module):
    """
    注意力分类损失

    组合：
    - 交叉熵损失（主损失）
    - 标签平滑（可选，AAD实验证明0.1有害，默认0）
    - 一致性正则化（可选，AAD中有用）

    Args:
        n_classes: 类别数
        label_smoothing: 标签平滑因子（默认0.0）
        use_consistency: 是否使用一致性正则化
        consistency_lambda: 一致性正则化权重
        class_weights: 类别权重（可选，用于类别不平衡）
    """

    def __init__(
        self,
        n_classes: int = 3,
        label_smoothing: float = 0.0,
        use_consistency: bool = False,
        consistency_lambda: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.use_consistency = use_consistency
        self.consistency_lambda = consistency_lambda

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subjects: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: 分类logits [batch, n_classes]
            labels: 标签 [batch]
            subjects: 被试ID [batch]（用于一致性正则化）

        Returns:
            字典，包含 loss 和各分项损失
        """
        # 主损失：交叉熵
        ce = self.ce_loss(logits, labels)

        total_loss = ce
        loss_dict = {"ce_loss": ce.item(), "total_loss": ce.item()}

        # 一致性正则化（可选）
        if self.use_consistency and subjects is not None and self.consistency_lambda > 0:
            consistency = self._compute_consistency_loss(logits, subjects)
            total_loss = total_loss + self.consistency_lambda * consistency
            loss_dict["consistency_loss"] = consistency.item()
            loss_dict["total_loss"] = total_loss.item()

        return {"loss": total_loss, **loss_dict}

    def _compute_consistency_loss(
        self,
        logits: torch.Tensor,
        subjects: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算一致性正则化损失

        思想：同一被试的不同窗口，预测概率分布应该一致。
        计算同一被试内所有窗口对的 KL 散度，取平均。

        Args:
            logits: [batch, n_classes]
            subjects: [batch]

        Returns:
            一致性损失（标量）
        """
        probs = F.softmax(logits, dim=-1)  # [batch, n_classes]
        unique_subjects = torch.unique(subjects)

        if len(unique_subjects) == len(subjects):
            # 每个被试只有一个样本，无法计算一致性
            return torch.tensor(0.0, device=logits.device)

        total_kl = 0.0
        count = 0

        for subj in unique_subjects:
            mask = subjects == subj
            if mask.sum() < 2:
                continue
            subj_probs = probs[mask]  # [n_subj_samples, n_classes]
            mean_prob = subj_probs.mean(dim=0, keepdim=True)  # [1, n_classes]

            # KL 散度：每个样本和均值的距离
            kl = F.kl_div(
                subj_probs.log(),
                mean_prob.expand_as(subj_probs),
                reduction="batchmean",
            )
            total_kl += kl
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device)

        return total_kl / count


class ConsistencyLoss(nn.Module):
    """
    独立的一致性正则化损失（可单独使用）

    和 AttentionLoss 中的一致性部分相同，但独立成类。
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        unique_subjects = torch.unique(subjects)

        if len(unique_subjects) == len(subjects):
            return torch.tensor(0.0, device=logits.device)

        total_kl = 0.0
        count = 0

        for subj in unique_subjects:
            mask = subjects == subj
            if mask.sum() < 2:
                continue
            subj_probs = probs[mask]
            mean_prob = subj_probs.mean(dim=0, keepdim=True)
            kl = F.kl_div(
                subj_probs.log(),
                mean_prob.expand_as(subj_probs),
                reduction="batchmean",
            )
            total_kl += kl
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device)

        return total_kl / count


class AAMPReconstructionLoss(nn.Module):
    """
    AAMP 预训练的重建损失

    只在被掩码的点上计算 L1 损失（和 NeurIPT 一致）。

    Args:
        loss_type: "l1" 或 "l2"
    """

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            reconstruction: 重建的信号 [batch, n_channels, time]
            original: 原始信号 [batch, n_channels, time]
            mask: 掩码标记 [batch, n_channels, time]，1=被掩码

        Returns:
            重建损失（只在被掩码的点上计算）
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=reconstruction.device)

        masked_recon = reconstruction[mask]
        masked_orig = original[mask]

        if self.loss_type == "l1":
            return F.l1_loss(masked_recon, masked_orig)
        elif self.loss_type == "l2":
            return F.mse_loss(masked_recon, masked_orig)
        else:
            raise ValueError(f"未知的损失类型: {self.loss_type}")
