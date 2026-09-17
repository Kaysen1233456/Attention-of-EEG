"""
全局配置类
借鉴 ms-swift 的配置驱动设计：所有参数集中管理，支持 YAML 读写，自动保存到实验目录
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import yaml
import json
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data/processed"
    sampling_rate: int = 250
    window_seconds: float = 2.0
    window_samples: int = 500  # 250Hz * 2s
    n_channels: int = 4
    channel_names: List[str] = field(default_factory=lambda: ["L04", "L05", "R04", "R05"])
    # 耳内4通道的三维相对坐标 (x:左右, y:前后, z:上下)
    channel_positions: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (-1.0, 0.5, 0.5),   # L04: 左耳偏前偏上
            (-1.0, 0.5, -0.5),  # L05: 左耳偏前偏下
            (1.0, 0.5, 0.5),    # R04: 右耳偏前偏上
            (1.0, 0.5, -0.5),   # R05: 右耳偏前偏下
        ]
    )
    # 左右耳分组（用于 IILP 左右耳分组池化）
    left_channel_indices: List[int] = field(default_factory=lambda: [0, 1])
    right_channel_indices: List[int] = field(default_factory=lambda: [2, 3])
    normalize: bool = True
    normalization_mode: str = "train_subjects_only"  # train_subjects_only / global
    bandpass_low: float = 0.5
    bandpass_high: float = 30.0
    electrode_coordinate_source: str = "provisional_relative_template"
    electrode_coordinate_frame: str = "ear_local_normalized_xyz"
    electrode_coordinate_version: str = "v0"


@dataclass
class ModelConfig:
    """模型配置"""
    architecture: str = "dual_branch"  # dual_branch / mini_neuript / single_branch
    d_model: int = 33  # 嵌入维度，必须能被3整除（3D坐标各占1/3），默认33
    # 双分支模型参数
    branch_conv1_out: int = 32
    branch_conv1_kernel: int = 5
    branch_conv1_stride: int = 2
    branch_conv2_out: int = 64
    branch_conv2_kernel: int = 3
    branch_conv2_stride: int = 2
    fusion_hidden: int = 64
    fusion_use_diff: bool = True       # 是否使用 |左耳-右耳| 特征
    fusion_use_product: bool = True    # 是否使用 左耳*右耳 特征
    # 分类头
    n_classes: int = 3  # 注意力3分类：深度专注/正常专注/分心
    dropout: float = 0.0  # 阶段一不用dropout（AAD实验证明有害）
    activation: str = "silu"  # silu(Swish) / gelu / relu，SwiGLU的门控基础
    # 3D电极嵌入
    use_3d_embedding: bool = True
    position_encoding_type: str = "sin_cos"  # sin_cos / learnable
    # 通道降维方式（3D嵌入后如何处理多通道）
    # none: 不平均，各通道各自过卷积（保留3D电极身份信息，推荐）
    # mean: 对通道维取平均（简单但抹掉通道差异）
    # flatten_linear: flatten后Linear投影回d_model
    channel_reduce: str = "none"  # none / mean / flatten_linear
    # 多层特征拼接
    use_multilayer_concat: bool = True
    use_iilp_pooling: bool = True
    use_pretrained_backbone: bool = False
    pretrained_backbone_path: Optional[str] = None
    freeze_pretrained_backbone: bool = False
    pretrained_n_heads: int = 8
    pretrained_n_layers: int = 4
    pretrained_d_ff: int = 384


@dataclass
class AAMPConfig:
    """AAMP 振幅感知掩码配置（用于自监督预训练）"""
    use_aamp: bool = True
    mask_ratio_range: List[float] = field(default_factory=lambda: [0.2, 0.35, 0.5])
    # BERT式掩码策略：80% [mask], 10% 随机, 10% 不变
    mask_token_ratio: float = 0.8
    random_token_ratio: float = 0.1
    unchanged_ratio: float = 0.1
    # 百分位采样范围
    percentile_low: float = 0.0
    percentile_high: float = 1.0
    # 幅值计算方式
    # abs: 按绝对值排序（EEG交流信号推荐，正负对称）
    # raw: 按原始信号值排序（严格复现论文字面表述）
    amplitude_type: str = "abs"  # abs / raw


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001883  # 从AAD Gen2.0迁移的最优lr
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # adamw / adam / sgd
    grad_clip_max_norm: float = 5.0
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"  # val_loss / val_accuracy / val_balanced_accuracy
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    use_amp: bool = True
    label_smoothing: float = 0.0  # AAD实验证明0.1有害，默认0
    # 一致性正则化（可选，AAD中有用）
    use_consistency_loss: bool = False
    consistency_lambda: float = 0.0
    # 学习率调度
    lr_scheduler: str = "none"  # none / cosine / step / onecycle
    warmup_ratio: float = 0.0


@dataclass
class SearchConfig:
    """超参数搜索配置"""
    search_method: str = "quasi_random"  # quasi_random / bayesian
    n_trials: int = 50
    # 准随机搜索
    quasi_random_method: str = "sobol"  # sobol / halton / latin_hypercube
    # 贝叶斯优化
    bayesian_acquisition: str = "ei"  # ei / ucb / poi
    bayesian_n_initial: int = 10
    bayesian_kernel: str = "matern"  # matern / rbf
    # 搜索空间（键=参数名，值=[低,高,类型]）
    search_space: dict = field(default_factory=lambda: {
        "learning_rate": [1e-4, 5e-3, "float"],
        "batch_size": [16, 64, "int"],
        "d_model": [24, 48, 72, 96],
        "branch_conv1_out": [16, 64, "int"],
        "branch_conv2_out": [32, 128, "int"],
        "fusion_hidden": [32, 128, "int"],
        "dropout": [0.0, 0.3, "float"],
    })
    # 目标指标
    objective_metric: str = "balanced_accuracy"  # 最大化（验证集指标）
    # 每次试验的训练轮数（搜索时可以减少epoch加速）
    search_epochs: int = 30


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: str = "artifacts/default"
    save_best_model: bool = True
    save_checkpoint_every: int = 0  # 0=只保存best
    log_interval: int = 50  # 每多少个batch打印一次
    use_tensorboard: bool = True
    save_config: bool = True  # 自动保存配置到实验目录（ms-swift风格）


@dataclass
class AttentionConfig:
    """总配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aamp: AAMPConfig = field(default_factory=AAMPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    project_name: str = "EEG-Attention-Decoding"
    experiment_name: str = "baseline_dual_branch"
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "AttentionConfig":
        """从YAML文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "AttentionConfig":
        """从JSON文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "AttentionConfig":
        """从字典构建配置（支持嵌套）"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key) and isinstance(value, dict):
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
            elif hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_dict(self) -> dict:
        """转换为字典（tuple转list，兼容YAML safe_load）"""
        def _convert(obj):
            if isinstance(obj, tuple):
                return [_convert(item) for item in obj]
            elif isinstance(obj, list):
                return [_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            else:
                return obj
        return _convert(asdict(self))

    def save_yaml(self, path: str):
        """保存为YAML（自动创建目录）"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def save_json(self, path: str):
        """保存为JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def update_from_dict(self, updates: dict):
        """从字典更新配置（用于超参数搜索）"""
        for key, value in updates.items():
            # 支持点号分隔的嵌套键，如 "training.learning_rate"
            if "." in key:
                parts = key.split(".")
                obj = self
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
            else:
                # 尝试在各子配置中查找
                for sub_name in ["data", "model", "aamp", "training", "search", "output"]:
                    sub = getattr(self, sub_name)
                    if hasattr(sub, key):
                        setattr(sub, key, value)
                        break
