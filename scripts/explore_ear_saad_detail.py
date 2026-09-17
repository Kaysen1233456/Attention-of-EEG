"""
深入探查 Ear-SAAD .mat 文件结构
查看每个trial的维度、通道名称、标签含义
"""
import scipy.io as sio
import numpy as np
from pathlib import Path

mat_path = Path(r"D:\Attention\data\raw\ear_saad\preprocessedData\preprocessedData\dataSubject1.mat")
mat = sio.loadmat(mat_path)

print("=" * 70)
print("1. 采样率")
print("=" * 70)
fs = mat['fs'][0][0]
print(f"  fs = {fs} Hz")

print("\n" + "=" * 70)
print("2. 通道名称 (61个通道)")
print("=" * 70)
channels = mat['channels']
print(f"  channels shape: {channels.shape}")
channel_names = []
for i in range(len(channels)):
    name = channels[i][0]
    if isinstance(name, np.ndarray):
        name = ''.join(name.flatten().tolist())
    channel_names.append(str(name))
    print(f"  [{i:2d}] {name}")

print(f"\n  总通道数: {len(channel_names)}")

# 分类通道
scalp_channels = []
around_ear_channels = []
in_ear_channels = []
eog_channels = []
other_channels = []

for i, name in enumerate(channel_names):
    name_lower = name.lower()
    if 'eog' in name_lower or 'vEOG' in name or 'hEOG' in name:
        eog_channels.append((i, name))
    elif name.startswith('L') and any(c.isdigit() for c in name):
        around_ear_channels.append((i, name))
    elif name.startswith('R') and any(c.isdigit() for c in name):
        around_ear_channels.append((i, name))
    elif 'ear' in name_lower or 'in-ear' in name_lower or 'TIP' in name:
        in_ear_channels.append((i, name))
    elif len(name) <= 4 and any(c.isdigit() for c in name):
        scalp_channels.append((i, name))
    else:
        other_channels.append((i, name))

print(f"\n  分类统计:")
print(f"    头皮EEG: {len(scalp_channels)} 通道")
print(f"    耳周EEG (around-ear): {len(around_ear_channels)} 通道")
print(f"    耳内EEG (in-ear): {len(in_ear_channels)} 通道")
print(f"    EOG: {len(eog_channels)} 通道")
print(f"    其他: {len(other_channels)} 通道")

if around_ear_channels:
    print(f"\n  耳周通道详情:")
    for i, name in around_ear_channels:
        print(f"    [{i:2d}] {name}")

if in_ear_channels:
    print(f"\n  耳内通道详情:")
    for i, name in in_ear_channels:
        print(f"    [{i:2d}] {name}")

if other_channels:
    print(f"\n  其他通道:")
    for i, name in other_channels:
        print(f"    [{i:2d}] {name}")

print("\n" + "=" * 70)
print("3. Trial 数据维度")
print("=" * 70)
eeg_trials = mat['eegTrials']
print(f"  eegTrials shape: {eeg_trials.shape}")
for i in range(len(eeg_trials)):
    trial_data = eeg_trials[i][0]
    if isinstance(trial_data, np.ndarray):
        print(f"  Trial {i+1}: shape={trial_data.shape}, dtype={trial_data.dtype}")
        if trial_data.ndim == 2:
            n_ch, n_time = trial_data.shape
            duration = n_time / fs
            print(f"           通道数={n_ch}, 时间点数={n_time}, 时长={duration:.1f}秒 ({duration/60:.1f}分钟)")
            print(f"           min={np.min(trial_data):.4f}, max={np.max(trial_data):.4f}, mean={np.mean(trial_data):.4f}")

print("\n" + "=" * 70)
print("4. 标签信息")
print("=" * 70)
att_speaker = mat['attSpeaker'].flatten()
attended_ear = mat['attendedEar'].flatten()
video_condition = mat['videoCondition'].flatten()

print(f"  attSpeaker (注意的说话人): {att_speaker}")
print(f"  attendedEar (注意的耳朵): {attended_ear}")
print(f"  videoCondition (视频条件): {video_condition}")
print(f"  attSpeaker == attendedEar: {np.array_equal(att_speaker, attended_ear)}")

# 统计标签分布
print(f"\n  attSpeaker 分布:")
unique, counts = np.unique(att_speaker, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    标签 {u}: {c} 个trial")

print("\n" + "=" * 70)
print("5. 语音包络 (envelopes)")
print("=" * 70)
envelopes = mat['envelopes']
print(f"  envelopes shape: {envelopes.shape}")
for i in range(len(envelopes)):
    env_data = envelopes[i][0]
    if isinstance(env_data, np.ndarray):
        print(f"  Trial {i+1}: shape={env_data.shape}, dtype={env_data.dtype}")
        if env_data.ndim == 2:
            print(f"           维度1={env_data.shape[0]}, 维度2={env_data.shape[1]}")

print("\n" + "=" * 70)
print("探查完成")
print("=" * 70)
