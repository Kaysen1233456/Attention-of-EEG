"""全面导入测试：确保所有模块都能正确导入"""
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("全面导入测试")
print("=" * 60)

# 1. 配置模块
print("\n1. 配置模块...")
from attention_model.config import (
    AttentionConfig, DataConfig, ModelConfig, AAMPConfig,
    TrainingConfig, SearchConfig, OutputConfig
)
print("   OK")

# 2. 模型模块
print("\n2. 模型模块...")
from attention_model.models import (
    Electrode3DEmbedding, DualBranchAttentionClassifier,
    MiniNeurIPT, SwiGLU, get_activation
)
print("   OK")

# 3. 数据模块
print("\n3. 数据模块...")
from attention_model.data import (
    AAMPMasking, EEGDataset, EEGWindowDataset,
    SyntheticEEGDataset, create_synthetic_dataloaders
)
print("   OK")

# 4. 训练模块
print("\n4. 训练模块...")
from attention_model.training import (
    AttentionTrainer, set_seed, AttentionLoss,
    ConsistencyLoss, AAMPReconstructionLoss
)
print("   OK")

# 5. 评估模块
print("\n5. 评估模块...")
from attention_model.evaluation import (
    compute_accuracy, compute_balanced_accuracy, compute_macro_f1,
    compute_roc_auc, compute_confusion_matrix,
    compute_subject_level_accuracy, evaluate_model
)
print("   OK")

# 6. 搜索模块
print("\n6. 搜索模块...")
from attention_model.search import QuasiRandomSearch, BayesianOptimization
print("   OK")

print("\n" + "=" * 60)
print("所有模块导入成功！")
print("=" * 60)

# 测试配置创建
print("\n配置测试:")
config = AttentionConfig()
print(f"  d_model: {config.model.d_model}")
print(f"  activation: {config.model.activation}")
print(f"  architecture: {config.model.architecture}")
print(f"  n_classes: {config.model.n_classes}")
print(f"  lr: {config.training.learning_rate}")
print(f"  batch_size: {config.training.batch_size}")
print(f"  seeds: {config.training.seeds}")

# 测试合成数据创建（不依赖torch）
print("\n合成数据测试（numpy部分）:")
import numpy as np
rng = np.random.RandomState(42)
waveforms = rng.randn(10, 4, 500).astype(np.float32)
labels = rng.randint(0, 3, size=10)
print(f"  waveforms shape: {waveforms.shape}")
print(f"  labels shape: {labels.shape}")
print(f"  labels distribution: {np.bincount(labels)}")

print("\n所有测试通过！")
