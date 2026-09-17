"""冒烟测试：验证所有模块能正常导入和运行"""
import sys
sys.path.insert(0, 'src')

import torch

# 测试1: 导入所有模块
print('=== 测试1: 导入模块 ===')
from attention_model.config import AttentionConfig
from attention_model.models import Electrode3DEmbedding, DualBranchAttentionClassifier, MiniNeurIPT
from attention_model.data import AAMPMasking, EEGDataset, EEGWindowDataset
from attention_model.training import AttentionTrainer, AttentionLoss, ConsistencyLoss
from attention_model.evaluation import compute_accuracy, compute_balanced_accuracy, compute_macro_f1
from attention_model.search import QuasiRandomSearch, BayesianOptimization
print('所有模块导入成功！')

# 测试2: 3D电极嵌入
print('\n=== 测试2: 3D电极嵌入 ===')
emb = Electrode3DEmbedding(d_model=30, n_channels=4)
x = torch.randn(2, 4, 500)
out = emb(x)
print(f'输入形状: {x.shape}, 输出形状: {out.shape}')
assert out.shape == (2, 4, 500, 30)
print('3D电极嵌入测试通过！')

# 测试3: AAMP掩码
print('\n=== 测试3: AAMP掩码 ===')
aamp = AAMPMasking(mask_ratio_range=[0.5])
masked_x, mask = aamp(x)
print(f'掩码比例: {mask.float().mean().item():.3f}')
assert masked_x.shape == x.shape
print('AAMP掩码测试通过！')

# 测试4: 双分支模型
print('\n=== 测试4: 双分支模型 ===')
model = DualBranchAttentionClassifier(n_channels=4, n_classes=3, d_model=30)
output = model(x)
logits = output['logits']
print(f'logits形状: {logits.shape}')
print(f'参数量: {model.count_parameters():,}')
assert logits.shape == (2, 3)
print('双分支模型测试通过！')

# 测试5: 配置类
print('\n=== 测试5: 配置类 ===')
config = AttentionConfig()
print(f'默认学习率: {config.training.learning_rate}')
print(f'默认batch_size: {config.training.batch_size}')
config.update_from_dict({'learning_rate': 0.001})
print(f'更新后学习率: {config.training.learning_rate}')
print('配置类测试通过！')

# 测试6: 准随机搜索
print('\n=== 测试6: 准随机搜索 ===')
search_space = {'lr': [1e-4, 1e-2, 'float'], 'bs': [16, 64, 'int']}
searcher = QuasiRandomSearch(search_space, n_trials=5, method='sobol')
params = searcher.get_trial_params(0)
print(f'试验0参数: {params}')
searcher.record_result(0, params, {'acc': 0.85})
print(f'最佳值: {searcher.best_value}')
print('准随机搜索测试通过！')

# 测试7: 迷你版NeurIPT
print('\n=== 测试7: 迷你版NeurIPT ===')
mini_model = MiniNeurIPT(d_model=60, n_heads=4, d_ff=256, n_layers=2, n_channels=4)
output = mini_model(x, apply_mask=True)
print(f'重建形状: {output["reconstruction"].shape}')
print(f'损失: {output["loss"].item():.4f}')
print(f'参数量: {mini_model.count_parameters():,}')
print('迷你版NeurIPT测试通过！')

# 测试8: 梯度流动
print('\n=== 测试8: 梯度流动 ===')
x_grad = torch.randn(2, 4, 500, requires_grad=True)
output = model(x_grad)
loss = output['logits'].sum()
loss.backward()
assert x_grad.grad is not None
print('梯度流动测试通过！')

print('\n' + '='*50)
print('所有冒烟测试通过！项目搭建完成！')
print('='*50)
