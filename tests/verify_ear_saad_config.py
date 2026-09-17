"""验证Ear-SAAD配置文件并估算参数量"""
import yaml
import sys

with open('configs/ear_saad.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print('=== 配置文件加载成功 ===')
print(f'项目: {config["project_name"]}')
print(f'实验: {config["experiment_name"]}')
print(f'通道数: {config["data"]["n_channels"]}')
print(f'采样率: {config["data"]["sampling_rate"]} Hz')
print(f'窗口: {config["data"]["window_seconds"]}s = {config["data"]["window_samples"]}点')
print(f'分类数: {config["model"]["n_classes"]}')
print(f'左耳通道: {len(config["data"]["left_channel_indices"])}个')
print(f'右耳通道: {len(config["data"]["right_channel_indices"])}个')
print(f'3D坐标数: {len(config["data"]["channel_positions"])}个')
print(f'd_model: {config["model"]["d_model"]}')
print(f'channel_reduce: {config["model"]["channel_reduce"]}')

# 估算参数量
d_model = config['model']['d_model']
n_ch_left = len(config['data']['left_channel_indices'])
n_ch_right = len(config['data']['right_channel_indices'])
conv1_out = config['model']['branch_conv1_out']
conv2_out = config['model']['branch_conv2_out']

if config['model']['channel_reduce'] == 'none':
    conv1_in_left = n_ch_left * d_model
    conv1_in_right = n_ch_right * d_model
else:
    conv1_in_left = d_model
    conv1_in_right = d_model

# 左耳分支参数量
params_left = conv1_in_left * conv1_out * 5 + conv1_out  # conv1
params_left += conv1_out * conv2_out * 3 + conv2_out  # conv2
# 右耳分支参数量
params_right = conv1_in_right * conv1_out * 5 + conv1_out
params_right += conv1_out * conv2_out * 3 + conv2_out

# 融合层输入维度
out_dim = conv1_out + conv2_out * 2  # conv1 GAP + conv2 mean + conv2 std
fusion_in = out_dim * 4  # 左耳 + 右耳 + |差| + 乘积
fusion_hidden = config['model']['fusion_hidden']
# SwiGLU: w1, w2, w3 三个矩阵
params_fusion = fusion_in * fusion_hidden * 3 + fusion_hidden * 3
params_fusion += fusion_hidden * config['model']['n_classes'] + config['model']['n_classes']

total = params_left + params_right + params_fusion
print('')
print('=== 参数量估算 ===')
print(f'左耳conv1输入: {conv1_in_left}通道 ({n_ch_left}通道 x {d_model}维)')
print(f'右耳conv1输入: {conv1_in_right}通道 ({n_ch_right}通道 x {d_model}维)')
print(f'左耳分支: ~{params_left:,}参数')
print(f'右耳分支: ~{params_right:,}参数')
print(f'融合层输入: {fusion_in}维 (out_dim={out_dim} x 4特征)')
print(f'融合+分类头: ~{params_fusion:,}参数')
print(f'总参数量(估算): ~{total:,}参数')
print('')
print(f'对比: 4通道时参数量=148,401')
if total > 200000:
    print(f'警告: 参数量较大，嵌入式部署可能需要优化')
    print(f'  可选方案1: d_model=12 (必须被3整除)')
    print(f'  可选方案2: channel_reduce=flatten_linear')
    print(f'  可选方案3: 减小conv1_out/conv2_out')
