"""
训练器模块

借鉴 ms-swift 的 Trainer 设计：
- 配置驱动（所有参数从 Config 读取）
- 多种子训练
- 早停
- AMP 混合精度
- 梯度裁剪
- 自动保存配置和最佳模型
- 被试级评估
- TensorBoard 日志

和 AAD 项目的训练脚本相比，这里重构为类，更易维护和扩展。
"""
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from tqdm import tqdm

from ..config import AttentionConfig
from ..evaluation.metrics import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_macro_f1,
    compute_roc_auc,
    compute_confusion_matrix,
    compute_subject_level_accuracy,
)
from .losses import AttentionLoss


def set_seed(seed: int):
    """设置随机种子（保证可复现）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttentionTrainer:
    """
    注意力模型训练器

    Args:
        model: 模型
        config: 配置对象
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 设备
    """

    def __init__(
        self,
        model: nn.Module,
        config: AttentionConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = self.model.to(self.device)

        # 损失函数
        self.criterion = AttentionLoss(
            n_classes=config.model.n_classes,
            label_smoothing=config.training.label_smoothing,
            use_consistency=config.training.use_consistency_loss,
            consistency_lambda=config.training.consistency_lambda,
        )

        # 优化器
        if config.training.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        elif config.training.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )

        # AMP
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # 学习率调度器
        self.scheduler = None
        if config.training.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.training.epochs
            )

        # 输出目录
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_metric = -float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_balanced_acc": [],
            "val_macro_f1": [],
        }

        # 保存配置（ms-swift 风格）
        if config.output.save_config:
            config.save_yaml(self.output_dir / "config.yaml")
            config.save_json(self.output_dir / "config.json")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}", leave=False)
        for batch in pbar:
            waveforms = batch["waveform"].to(self.device)
            labels = batch["label"].to(self.device)
            subjects = batch.get("subject", None)
            if subjects is not None:
                subjects = subjects.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output = self.model(waveforms)
                logits = output["logits"]
                loss_dict = self.criterion(logits, labels, subjects)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            # 梯度裁剪
            if self.config.training.grad_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip_max_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 统计
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{total_correct/total_samples:.4f}",
            })

        return {
            "loss": total_loss / n_batches,
            "accuracy": total_correct / total_samples,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_logits = []
        all_labels = []
        all_subjects = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            waveforms = batch["waveform"].to(self.device)
            labels = batch["label"].to(self.device)
            subjects = batch.get("subject", None)
            if subjects is not None:
                subjects = subjects.to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(waveforms)
                logits = output["logits"]
                loss_dict = self.criterion(logits, labels, subjects)

            total_loss += loss_dict["loss"].item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if subjects is not None:
                all_subjects.append(subjects.cpu())
            n_batches += 1

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probs = torch.softmax(all_logits, dim=-1)

        metrics = {
            "loss": total_loss / n_batches,
            "accuracy": compute_accuracy(all_logits, all_labels),
            "balanced_accuracy": compute_balanced_accuracy(all_logits, all_labels),
            "macro_f1": compute_macro_f1(all_logits, all_labels),
        }

        # ROC-AUC（二分类或多分类都支持）
        try:
            metrics["roc_auc"] = compute_roc_auc(all_probs, all_labels)
        except Exception:
            metrics["roc_auc"] = 0.0

        # 混淆矩阵
        try:
            metrics["confusion_matrix"] = compute_confusion_matrix(all_logits, all_labels).tolist()
        except Exception:
            pass

        # 被试级评估（如果有被试信息）
        if len(all_subjects) > 0:
            all_subjects = torch.cat(all_subjects, dim=0)
            try:
                subj_metrics = compute_subject_level_accuracy(
                    all_probs, all_labels, all_subjects,
                    n_classes=self.config.model.n_classes,
                )
                metrics.update(subj_metrics)
            except Exception as e:
                print(f"被试级评估失败: {e}")

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        完整训练流程

        Returns:
            训练结果字典，包含历史记录和最佳指标
        """
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"输出目录: {self.output_dir}")

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.evaluate(self.val_loader)

            # 记录历史
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_balanced_acc"].append(val_metrics["balanced_accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            # 打印
            print(
                f"Epoch {epoch+1}/{self.config.training.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"BalAcc: {val_metrics['balanced_accuracy']:.4f} F1: {val_metrics['macro_f1']:.4f}"
            )

            # 早停判断（基于配置的指标，默认 val_loss）
            early_stopping_metric = self.config.training.early_stopping_metric
            # 映射配置名到 val_metrics 中的键
            metric_key_map = {
                "val_loss": "loss",
                "val_accuracy": "accuracy",
                "val_balanced_accuracy": "balanced_accuracy",
                "val_macro_f1": "macro_f1",
            }
            metric_key = metric_key_map.get(early_stopping_metric, "loss")
            current_metric = val_metrics[metric_key]

            # 判断是否最佳（loss 是越小越好，其他指标是越大越好）
            if early_stopping_metric == "val_loss":
                is_best = current_metric < self.best_val_loss
                if is_best:
                    self.best_val_loss = current_metric
            else:
                is_best = current_metric > self.best_val_metric
                if is_best:
                    self.best_val_metric = current_metric

            if is_best:
                self.patience_counter = 0
                # 保存最佳模型
                if self.config.output.save_best_model:
                    torch.save(
                        self.model.state_dict(),
                        self.output_dir / "best_model.pt",
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"早停触发！连续 {self.config.training.early_stopping_patience} 个epoch {early_stopping_metric} 没有改善")
                    break

        # 加载最佳模型
        if self.config.output.save_best_model and (self.output_dir / "best_model.pt").exists():
            self.model.load_state_dict(torch.load(self.output_dir / "best_model.pt", map_location=self.device))
            print("已加载最佳模型")

        # 测试集评估
        test_metrics = None
        if self.test_loader is not None:
            print("在测试集上评估...")
            test_metrics = self.evaluate(self.test_loader)
            print(f"测试集结果: Acc={test_metrics['accuracy']:.4f}, "
                  f"BalAcc={test_metrics['balanced_accuracy']:.4f}, "
                  f"F1={test_metrics['macro_f1']:.4f}")

        # 保存训练历史
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # 保存最终结果
        results = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.current_epoch + 1 - self.patience_counter,
            "final_val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_info": getattr(self.model, "get_model_info", lambda: {})(),
        }
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def train_multi_seed(self) -> Dict[str, Any]:
        """
        多种子训练（对每个种子训练一次，然后汇总）

        Returns:
            多种子汇总结果
        """
        all_results = []
        seeds = self.config.training.seeds

        for i, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"种子 {i+1}/{len(seeds)}: seed={seed}")
            print(f"{'='*60}")

            set_seed(seed)

            # 重置模型
            for layer in self.model.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

            # 重置优化器
            if self.config.training.optimizer == "adamw":
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                )

            # 重置训练状态
            self.current_epoch = 0
            self.best_val_loss = float("inf")
            self.patience_counter = 0
            self.history = {k: [] for k in self.history}

            # 为每个种子创建子目录
            original_output = self.output_dir
            self.output_dir = original_output / f"seed_{seed}"
            self.output_dir.mkdir(parents=True, exist_ok=True)

            result = self.train()
            all_results.append(result)

            self.output_dir = original_output

        # 汇总多种子结果
        summary = self._summarize_multi_seed(all_results, seeds)

        # 保存汇总
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _summarize_multi_seed(
        self,
        all_results: List[Dict],
        seeds: List[int],
    ) -> Dict[str, Any]:
        """汇总多种子结果"""
        summary = {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "per_seed": [],
        }

        # 收集每个种子的测试指标
        metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
        metric_values = {k: [] for k in metric_keys}

        for i, result in enumerate(all_results):
            seed_info = {"seed": seeds[i]}
            if result.get("test_metrics"):
                for k in metric_keys:
                    if k in result["test_metrics"]:
                        val = result["test_metrics"][k]
                        seed_info[k] = val
                        metric_values[k].append(val)
                # 被试级指标
                for k in result["test_metrics"]:
                    if "subject" in k.lower() or "subject_level" in k.lower():
                        seed_info[k] = result["test_metrics"][k]
            summary["per_seed"].append(seed_info)

        # 计算均值和标准差
        summary["mean"] = {}
        summary["std"] = {}
        for k in metric_keys:
            if len(metric_values[k]) > 0:
                summary["mean"][k] = float(np.mean(metric_values[k]))
                summary["std"][k] = float(np.std(metric_values[k]))

        return summary
