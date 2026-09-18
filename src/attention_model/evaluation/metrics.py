"""
评估指标模块

包含：
1. 窗口级指标：Accuracy, Balanced Accuracy, Macro F1, ROC-AUC, Confusion Matrix
2. 被试级指标：Subject-level Accuracy（聚合每个被试的所有窗口预测）
3. 综合评估函数：evaluate_model
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算准确率"""
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    return float(accuracy_score(labels_np, preds))


def compute_balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算平衡准确率（对类别不平衡不敏感）"""
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    return float(balanced_accuracy_score(labels_np, preds))


def compute_macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算 Macro F1"""
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    return float(f1_score(labels_np, preds, average="macro", zero_division=0))


def compute_roc_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算 ROC-AUC

    二分类：直接计算
    多分类：one-vs-rest，macro 平均
    """
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    n_classes = probs_np.shape[1]

    if n_classes == 2:
        value = roc_auc_score(labels_np, probs_np[:, 1])
    else:
        # 多分类 one-vs-rest
        value = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"ROC-AUC is non-finite: {value}")
    return value


def compute_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """计算混淆矩阵"""
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    return confusion_matrix(labels_np, preds)


def compute_subject_level_accuracy(
    probs: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    n_classes: int = 3,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算被试级准确率

    思想：对每个被试的所有窗口预测概率取平均，再用阈值判断该被试的类别。
    这是 AAD 项目验证有效的评估方式，因为每个被试的所有窗口共享同一标签。

    Args:
        probs: 预测概率 [N, n_classes]
        labels: 标签 [N]
        subjects: 被试ID [N]
        n_classes: 类别数
        threshold: 二分类阈值（仅二分类时用）

    Returns:
        被试级指标字典
    """
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    subjects_np = subjects.cpu().numpy()

    unique_subjects = np.unique(subjects_np)
    subject_predictions = []
    subject_labels = []
    subject_avg_probs = []

    for subj in unique_subjects:
        mask = subjects_np == subj
        subj_probs = probs_np[mask]  # [n_windows, n_classes]
        unique_labels = np.unique(labels_np[mask])
        if len(unique_labels) != 1:
            raise ValueError(
                "subject-level aggregation requires one label per subject; "
                f"subject={subj} has labels {unique_labels.tolist()}"
            )
        subj_label = unique_labels[0]

        # 对所有窗口的概率取平均
        avg_prob = subj_probs.mean(axis=0)  # [n_classes]
        subject_avg_probs.append(avg_prob)

        # 预测类别
        if n_classes == 2:
            pred = 1 if avg_prob[1] > threshold else 0
        else:
            pred = int(np.argmax(avg_prob))

        subject_predictions.append(pred)
        subject_labels.append(subj_label)

    subject_predictions = np.array(subject_predictions)
    subject_labels = np.array(subject_labels)
    subject_avg_probs = np.array(subject_avg_probs)

    # 计算指标
    accuracy = float(accuracy_score(subject_labels, subject_predictions))
    balanced_acc = float(balanced_accuracy_score(subject_labels, subject_predictions))

    # 最优阈值搜索（二分类时）
    best_threshold_acc = accuracy
    best_threshold = threshold
    if n_classes == 2:
        for t in np.arange(0.1, 0.9, 0.01):
            preds_t = (subject_avg_probs[:, 1] > t).astype(int)
            acc_t = accuracy_score(subject_labels, preds_t)
            if acc_t > best_threshold_acc:
                best_threshold_acc = acc_t
                best_threshold = t

    return {
        "subject_level_accuracy": accuracy,
        "subject_level_balanced_accuracy": balanced_acc,
        "subject_level_n_subjects": len(unique_subjects),
        "subject_level_correct": int((subject_predictions == subject_labels).sum()),
        "subject_level_best_threshold": float(best_threshold),
        "subject_level_best_threshold_accuracy": float(best_threshold_acc),
    }


def compute_subject_trial_accuracy(
    probs: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    trials: torch.Tensor,
) -> Dict[str, float]:
    """Aggregate windows by (subject, trial), requiring one label per group."""
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    subjects_np = subjects.cpu().numpy()
    trials_np = trials.cpu().numpy()
    keys = np.stack([subjects_np, trials_np], axis=1)
    group_probs = []
    group_labels = []
    for subject_id, trial_id in np.unique(keys, axis=0):
        mask = (subjects_np == subject_id) & (trials_np == trial_id)
        unique_labels = np.unique(labels_np[mask])
        if len(unique_labels) != 1:
            raise ValueError(
                f"标签在 subject={subject_id}, trial={trial_id} 内不一致: {unique_labels.tolist()}"
            )
        group_probs.append(probs_np[mask].mean(axis=0))
        group_labels.append(unique_labels[0])
    group_probs = np.asarray(group_probs)
    group_labels = np.asarray(group_labels)
    predictions = group_probs.argmax(axis=1)
    return {
        "subject_trial_accuracy": float(accuracy_score(group_labels, predictions)),
        "subject_trial_balanced_accuracy": float(
            balanced_accuracy_score(group_labels, predictions)
        ),
        "subject_trial_n_groups": int(len(group_labels)),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_classes: int = 3,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    综合评估模型

    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        n_classes: 类别数
        use_amp: 是否使用AMP

    Returns:
        评估指标字典
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_subjects = []
    all_trials = []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            subjects = batch.get("subject", None)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                output = model(waveforms)
                logits = output["logits"]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if subjects is not None:
                all_subjects.append(subjects.cpu())
            trials = batch.get("trial", None)
            if trials is not None:
                all_trials.append(trials.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.softmax(all_logits, dim=-1)

    metrics = {
        "accuracy": compute_accuracy(all_logits, all_labels),
        "balanced_accuracy": compute_balanced_accuracy(all_logits, all_labels),
        "macro_f1": compute_macro_f1(all_logits, all_labels),
        "roc_auc": compute_roc_auc(all_probs, all_labels),
        "confusion_matrix": compute_confusion_matrix(all_logits, all_labels).tolist(),
        "n_samples": len(all_labels),
    }

    # 被试级评估
    if len(all_subjects) > 0 and len(all_trials) > 0:
        all_subjects = torch.cat(all_subjects, dim=0)
        all_trials = torch.cat(all_trials, dim=0)
        try:
            subj_metrics = compute_subject_trial_accuracy(
                all_probs, all_labels, all_subjects, all_trials,
            )
            metrics.update(subj_metrics)
        except Exception as e:
            print(f"被试级评估失败: {e}")

    return metrics
