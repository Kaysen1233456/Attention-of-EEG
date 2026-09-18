"""Losses and projection heads used for teacher-student distillation."""

import torch.nn as nn
import torch.nn.functional as F


class BilateralFeatureProjector(nn.Module):
    """Map student left/right features into the teacher feature space."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.left = nn.Sequential(nn.LayerNorm(student_dim), nn.Linear(student_dim, teacher_dim))
        self.right = nn.Sequential(nn.LayerNorm(student_dim), nn.Linear(student_dim, teacher_dim))

    def forward(self, left_features, right_features):
        return self.left(left_features), self.right(right_features)


class DistillationLoss(nn.Module):
    """Combine hard labels, teacher logits, and bilateral feature matching."""

    def __init__(self, temperature=2.0, label_weight=1.0, logit_weight=1.0, feature_weight=0.25):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.label_weight = label_weight
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight

    def forward(
        self,
        student_logits,
        labels,
        teacher_logits,
        student_left,
        student_right,
        teacher_left,
        teacher_right,
        projector,
    ):
        hard_loss = F.cross_entropy(student_logits, labels)
        temperature = self.temperature
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        projected_left, projected_right = projector(student_left, student_right)
        feature_loss = 0.5 * (
            F.smooth_l1_loss(projected_left, teacher_left.detach())
            + F.smooth_l1_loss(projected_right, teacher_right.detach())
        )
        total = (
            self.label_weight * hard_loss
            + self.logit_weight * soft_loss
            + self.feature_weight * feature_loss
        )
        return {
            "loss": total,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "feature_loss": feature_loss,
        }
