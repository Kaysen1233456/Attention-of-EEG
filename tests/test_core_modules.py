"""非torch依赖模块测试：配置模块 + 搜索算法"""
import sys
sys.path.insert(0, 'src')

# 测试配置模块
from attention_model.config import AttentionConfig
config = AttentionConfig()
print('配置模块测试:')
print(f'  d_model: {config.model.d_model}')
print(f'  activation: {config.model.activation}')
print(f'  lr: {config.training.learning_rate}')
print(f'  配置保存/加载测试...')
config.save_yaml('tests/test_config.yaml')
config2 = AttentionConfig.from_yaml('tests/test_config.yaml')
assert config2.model.d_model == config.model.d_model
print('  配置保存/加载: OK')

# 测试准随机搜索
from attention_model.search.quasi_random import QuasiRandomSearch
search_space = {
    'lr': (1e-4, 5e-3, 'float'),
    'batch_size': (16, 64, 'int'),
}
search = QuasiRandomSearch(search_space, n_trials=10, method='sobol')
print(f'准随机搜索测试: 共{search.n_trials}个试验点')
first_params = search.get_trial_params(0)
print(f'  第一个点: lr={first_params["lr"]:.6f}, batch_size={first_params["batch_size"]}')

# 测试贝叶斯优化
from attention_model.search.bayesian_optimization import BayesianOptimization
bo = BayesianOptimization(search_space, n_initial=5, acquisition='ei', maximize=True)
for i in range(5):
    params = bo.get_next_params()
    score = 0.5 + 0.1 * i  # 模拟分数
    bo.record_result(params, {'score': score})
next_params = bo.get_next_params()
print(f'贝叶斯优化测试: 建议下一个点 lr={next_params["lr"]:.6f}')

print()
print('所有非torch依赖模块测试通过！')
