"""
数据集和数据加载模块

支持的数据格式（和 AAD 项目一致）：
- 预处理后的 npy 文件：train_waveforms.npy, train_labels.npy, train_subjects.npy
- 形状：[N_samples, n_channels, time_points]
- 标签：0/1 二分类 或 多分类

支持的功能：
- 窗口级数据集
- 被试级划分（train/val/test 被试不重叠）
- 数据归一化（训练集统计量）
- 数据增强（可选）
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class EEGWindowDataset(Dataset):
    """
    EEG 窗口级数据集

    每个样本是一个 EEG 窗口：[n_channels, time_points]
    标签是一个整数（分类任务）或浮点数（回归任务）

    Args:
        waveforms: EEG 数据 [N, n_channels, time_points]
        labels: 标签 [N]（分类任务为整数，回归任务为浮点数）
        subjects: 被试ID [N]（可选，用于被试级评估）
        transform: 数据增强变换（可选）
        normalize: 是否归一化
        mean: 归一化均值 [n_channels, 1]（如果提供则用这个，否则从数据计算）
        std: 归一化标准差 [n_channels, 1]
    """

    def __init__(
        self,
        waveforms: np.ndarray,
        labels: np.ndarray,
        subjects: Optional[np.ndarray] = None,
        transform: Optional[Any] = None,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        assert waveforms.ndim == 3, f"waveforms 应该是3维 [N, C, T]，当前是 {waveforms.ndim} 维"
        assert len(waveforms) == len(labels), f"样本数不匹配: {len(waveforms)} vs {len(labels)}"

        self.waveforms = waveforms.astype(np.float32)
        self.labels = labels
        self.subjects = subjects
        self.transform = transform
        self.normalize = normalize

        # 计算归一化统计量
        if normalize:
            if mean is not None and std is not None:
                self.mean = mean.astype(np.float32)
                self.std = std.astype(np.float32)
            else:
                # 从数据计算（按通道）
                self.mean = waveforms.mean(axis=(0, 2), keepdims=True).squeeze(0)  # [C, 1]
                self.std = waveforms.std(axis=(0, 2), keepdims=True).squeeze(0) + 1e-8  # [C, 1]
        else:
            self.mean = None
            self.std = None

        self.n_channels = waveforms.shape[1]
        self.time_points = waveforms.shape[2]

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        waveform = self.waveforms[idx]  # [C, T]
        label = self.labels[idx]

        # 归一化
        if self.normalize and self.mean is not None:
            waveform = (waveform - self.mean) / self.std

        # 数据增强
        if self.transform is not None:
            waveform = self.transform(waveform)

        # 转 tensor
        waveform = torch.from_numpy(waveform.copy()).float()

        if isinstance(label, (int, np.integer)):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float)

        output = {
            "waveform": waveform,
            "label": label,
            "index": idx,
        }

        if self.subjects is not None:
            output["subject"] = self.subjects[idx]

        return output

    def get_label_distribution(self) -> Dict:
        """获取标签分布"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def get_subject_ids(self) -> List:
        """获取所有被试ID"""
        if self.subjects is None:
            return []
        return list(np.unique(self.subjects))


class EEGDataset:
    """
    完整的 EEG 数据集（包含 train/val/test 划分）

    从预处理后的 npy 文件加载，支持被试级划分。

    数据目录结构（和 AAD 项目一致）：
        data_dir/
        ├── train_waveforms.npy   [N_train, C, T]
        ├── train_labels.npy      [N_train]
        ├── train_subjects.npy    [N_train]
        ├── val_waveforms.npy     [N_val, C, T]
        ├── val_labels.npy        [N_val]
        ├── val_subjects.npy      [N_val]
        ├── test_waveforms.npy    [N_test, C, T]
        ├── test_labels.npy       [N_test]
        ├── test_subjects.npy     [N_test]
        └── normalization.npz     (可选，包含 mean 和 std)

    Args:
        data_dir: 数据目录
        normalize: 是否归一化
        use_train_stats: 是否用训练集统计量做归一化（推荐True，避免数据泄漏）
    """

    def __init__(
        self,
        data_dir: str,
        normalize: bool = True,
        use_train_stats: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.use_train_stats = use_train_stats

        # 加载数据
        self._load_data()

        # 计算归一化统计量（用训练集）
        if normalize and use_train_stats:
            self.mean = self.train_waveforms.mean(axis=(0, 2), keepdims=True).squeeze(0)  # [C, 1]
            self.std = self.train_waveforms.std(axis=(0, 2), keepdims=True).squeeze(0) + 1e-8
        else:
            self.mean = None
            self.std = None

        # 创建数据集
        self.train_dataset = EEGWindowDataset(
            self.train_waveforms, self.train_labels, self.train_subjects,
            normalize=normalize, mean=self.mean, std=self.std,
        )
        self.val_dataset = EEGWindowDataset(
            self.val_waveforms, self.val_labels, self.val_subjects,
            normalize=normalize, mean=self.mean, std=self.std,
        )
        self.test_dataset = EEGWindowDataset(
            self.test_waveforms, self.test_labels, self.test_subjects,
            normalize=normalize, mean=self.mean, std=self.std,
        )

    def _load_data(self):
        """加载 npy 数据"""
        # 训练集
        self.train_waveforms = np.load(self.data_dir / "train_waveforms.npy")
        self.train_labels = np.load(self.data_dir / "train_labels.npy")
        train_subj_path = self.data_dir / "train_subjects.npy"
        self.train_subjects = np.load(train_subj_path) if train_subj_path.exists() else None

        # 验证集
        val_wave_path = self.data_dir / "val_waveforms.npy"
        if val_wave_path.exists():
            self.val_waveforms = np.load(val_wave_path)
            self.val_labels = np.load(self.data_dir / "val_labels.npy")
            val_subj_path = self.data_dir / "val_subjects.npy"
            self.val_subjects = np.load(val_subj_path) if val_subj_path.exists() else None
        else:
            # 没有验证集，从训练集分 10%
            n = len(self.train_waveforms)
            n_val = int(n * 0.1)
            indices = np.random.permutation(n)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            self.val_waveforms = self.train_waveforms[val_idx]
            self.val_labels = self.train_labels[val_idx]
            self.val_subjects = self.train_subjects[val_idx] if self.train_subjects is not None else None
            self.train_waveforms = self.train_waveforms[train_idx]
            self.train_labels = self.train_labels[train_idx]
            self.train_subjects = self.train_subjects[train_idx] if self.train_subjects is not None else None

        # 测试集
        test_wave_path = self.data_dir / "test_waveforms.npy"
        if test_wave_path.exists():
            self.test_waveforms = np.load(test_wave_path)
            self.test_labels = np.load(self.data_dir / "test_labels.npy")
            test_subj_path = self.data_dir / "test_subjects.npy"
            self.test_subjects = np.load(test_subj_path) if test_subj_path.exists() else None
        else:
            # 没有测试集，用验证集代替
            self.test_waveforms = self.val_waveforms
            self.test_labels = self.val_labels
            self.test_subjects = self.val_subjects

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
    ) -> Dict[str, DataLoader]:
        """
        获取 train/val/test 的 DataLoader

        Args:
            batch_size: batch 大小
            num_workers: 数据加载线程数
            shuffle_train: 是否打乱训练集

        Returns:
            字典，包含 train/val/test 三个 DataLoader
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

    def get_info(self) -> Dict:
        """获取数据集信息"""
        return {
            "data_dir": str(self.data_dir),
            "n_train": len(self.train_dataset),
            "n_val": len(self.val_dataset),
            "n_test": len(self.test_dataset),
            "n_channels": self.train_dataset.n_channels,
            "time_points": self.train_dataset.time_points,
            "train_label_dist": self.train_dataset.get_label_distribution(),
            "val_label_dist": self.val_dataset.get_label_distribution(),
            "test_label_dist": self.test_dataset.get_label_distribution(),
            "train_subjects": self.train_dataset.get_subject_ids(),
            "val_subjects": self.val_dataset.get_subject_ids(),
            "test_subjects": self.test_dataset.get_subject_ids(),
        }


class SyntheticEEGDataset(Dataset):
    """
    合成 EEG 数据集（用于测试和冒烟测试）

    生成随机 EEG 信号和随机标签，用于在没有真实数据时测试代码。

    Args:
        n_samples: 样本数
        n_channels: 通道数
        time_points: 时间点数
        n_classes: 类别数
        seed: 随机种子
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_channels: int = 4,
        time_points: int = 500,
        n_classes: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.waveforms = rng.randn(n_samples, n_channels, time_points).astype(np.float32)
        self.labels = rng.randint(0, n_classes, size=n_samples)
        self.subjects = rng.randint(0, 10, size=n_samples)  # 10个虚拟被试

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "waveform": torch.from_numpy(self.waveforms[idx]).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "subject": self.subjects[idx],
            "index": idx,
        }


def create_synthetic_dataloaders(
    batch_size: int = 32,
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    n_channels: int = 4,
    time_points: int = 500,
    n_classes: int = 3,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    创建合成数据的 DataLoader（用于冒烟测试）

    Returns:
        字典，包含 train/val/test 三个 DataLoader
    """
    train_ds = SyntheticEEGDataset(n_train, n_channels, time_points, n_classes, seed)
    val_ds = SyntheticEEGDataset(n_val, n_channels, time_points, n_classes, seed + 1)
    test_ds = SyntheticEEGDataset(n_test, n_channels, time_points, n_classes, seed + 2)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
