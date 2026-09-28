"""
心算专注数据集预处理脚本（250Hz版本）

功能：
1. 读取36个被试的EDF文件（每人2个：静息 + 心算专注）
2. 选择8个重点电极（Fz/F3/F4/T7/T8/Pz/P3/P4）
3. 带通滤波 1-45Hz
4. 降采样 500Hz → 250Hz
5. 切2秒窗口（500采样点）
6. 二分类标签：静息(0) vs 专注(1)
7. 按被试划分 train/val/test
8. 计算精确的 class_weights（方法A：简单反比）
9. 保存为 .npy 文件

用法：
    python scripts/preprocess_mental_arithmetic.py
"""

import numpy as np
import mne
import os
import json
from pathlib import Path
from collections import Counter

# ==================== 配置 ====================

# 输入路径（EDF原始数据所在目录）
INPUT_DIR = Path("/mnt/workspace/attention_data/eegmat")  # 你的数据集实际路径

# 输出路径
OUTPUT_DIR = Path("data/processed/mental_arithmetic_250hz_8ch")

# EDF文件命名规则
# Subject00_1.edf = 静息态
# Subject00_2.edf = 专注态（心算）
REST_SUFFIX = "_1.edf"
FOCUS_SUFFIX = "_2.edf"

# 被试编号范围
SUBJECT_START = 0
SUBJECT_END = 35  # Subject00 到 Subject35，共36人

# EDF里的19个EEG通道顺序（精确顺序）
EDF_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8",
    "T3", "T4", "C3", "C4", "T5", "T6",
    "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"
]

# 要选择的8个重点电极（在EDF里的名称）
# T3=T7, T4=T8（旧命名映射）
SELECTED_CHANNELS_IN_EDF = ["Fz", "F3", "F4", "T3", "T4", "Pz", "P3", "P4"]

# 输出的通道名称（用新命名）
SELECTED_CHANNEL_NAMES = ["Fz", "F3", "F4", "T7", "T8", "Pz", "P3", "P4"]

# 8个通道的10-20系统近似3D坐标 (x, y, z)
# 用于3D电极嵌入
CHANNEL_POSITIONS = [
    [0.0, 1.0, 0.3],    # Fz
    [-0.4, 0.8, 0.5],   # F3
    [0.4, 0.8, 0.5],    # F4
    [-1.0, 0.0, 0.0],   # T7
    [1.0, 0.0, 0.0],    # T8
    [0.0, -1.0, 0.3],   # Pz
    [-0.4, -0.8, 0.5],  # P3
    [0.4, -0.8, 0.5],   # P4
]

# 预处理参数
RAW_SAMPLING_RATE = 500  # EDF原始采样率
TARGET_SAMPLING_RATE = 250  # 目标采样率
WINDOW_SECONDS = 2.0  # 窗口长度（秒）
WINDOW_SAMPLES = int(TARGET_SAMPLING_RATE * WINDOW_SECONDS)  # 500
BANDPASS_LOW = 1.0  # 带通滤波低频截止
BANDPASS_HIGH = 45.0  # 带通滤波高频截止

# 数据集划分（按被试）
# 36人：28训练 / 4验证 / 4测试
N_TRAIN_SUBJECTS = 28
N_VAL_SUBJECTS = 4
N_TEST_SUBJECTS = 4

# 随机种子（保证划分可复现）
RANDOM_SEED = 42


# ==================== 工具函数 ====================

def compute_class_weights(y_train, method="inverse"):
    """
    精确计算类别权重（方法A：简单反比）
    
    公式：weight_i = 总样本数 / (类别数 × 该类别样本数)
    """
    n_classes = len(np.unique(y_train))
    class_counts = np.bincount(y_train, minlength=n_classes)
    total = class_counts.sum()
    
    if method == "inverse":
        weights = total / (n_classes * class_counts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print("\n" + "=" * 60)
    print("类别权重计算（方法A：简单反比）")
    print("=" * 60)
    print(f"训练集总样本数：{total}")
    print(f"类别数：{n_classes}")
    print(f"各类别样本数：静息={class_counts[0]}, 专注={class_counts[1]}")
    print(f"各类别比例：静息={class_counts[0]/total:.4f}, 专注={class_counts[1]/total:.4f}")
    print(f"比例（静息:专注）= {class_counts[0]/class_counts[1]:.4f}:1")
    print(f"计算得到的权重：静息={weights[0]:.4f}, 专注={weights[1]:.4f}")
    print(f"权重比（专注/静息）= {weights[1]/weights[0]:.4f}")
    print("=" * 60 + "\n")
    
    return weights.tolist()


def process_one_edf(edf_path, label):
    """
    处理单个EDF文件
    
    返回：
        windows: np.ndarray, 形状 (n_windows, 8, 500)
        labels: np.ndarray, 形状 (n_windows,)
    """
    try:
        # 读取EDF
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"  ❌ 读取失败：{edf_path.name}")
        print(f"     错误：{e}")
        print(f"     跳过此文件")
        return np.array([]), np.array([])
    
    # 检查数据时长
    duration = raw.n_times / raw.info['sfreq']
    if duration < 10:  # 少于10秒，数据太少
        print(f"  ⚠️  文件过短：{edf_path.name}，只有 {duration:.1f} 秒")
        print(f"     仍然尝试处理，但窗口数会很少")
    
    # 选择8个EEG通道（EDF里的通道名可能带前缀，比如"EEG Fz"）
    # 先获取所有通道名
    all_ch_names = raw.ch_names
    
    # 找到我们要的通道（模糊匹配）
    picks = []
    for target_ch in SELECTED_CHANNELS_IN_EDF:
        found = False
        for ch_name in all_ch_names:
            # 去掉空格和前缀后比较
            clean_name = ch_name.replace(" ", "").replace("EEG", "").upper()
            if target_ch.upper() in clean_name:
                picks.append(ch_name)
                found = True
                break
        if not found:
            print(f"  警告：未找到通道 {target_ch}")
    
    if len(picks) < 8:
        print(f"  警告：只找到 {len(picks)}/8 个通道")
    
    if len(picks) == 0:
        print(f"  ❌ 没有找到任何EEG通道，跳过此文件")
        return np.array([]), np.array([])
    
    # 选择通道
    raw.pick(picks)
    
    # 带通滤波
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, verbose=False)
    
    # 降采样 500Hz → 250Hz
    raw.resample(TARGET_SAMPLING_RATE, verbose=False)
    
    # 获取数据 (n_channels, n_times)
    data = raw.get_data()
    
    # 切窗口
    n_channels, n_times = data.shape
    n_windows = n_times // WINDOW_SAMPLES
    
    if n_windows == 0:
        print(f"  ⚠️  数据不足一个窗口（{n_times} 采样点），跳过")
        return np.array([]), np.array([])
    
    windows = []
    for i in range(n_windows):
        start = i * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        window = data[:, start:end]
        windows.append(window)
    
    windows = np.array(windows)  # (n_windows, 8, 500)
    labels = np.full(n_windows, label)
    
    return windows, labels


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("心算专注数据集预处理（250Hz，8通道，2秒窗）")
    print("=" * 60)
    
    # 检查输入目录
    if not INPUT_DIR.exists():
        print(f"\n错误：输入目录不存在 {INPUT_DIR}")
        print("请把EDF文件放到这个目录下，或者修改脚本里的INPUT_DIR")
        return
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成被试列表
    all_subjects = [f"Subject{i:02d}" for i in range(SUBJECT_START, SUBJECT_END + 1)]
    print(f"\n总被试数：{len(all_subjects)}")
    
    # 随机打乱并划分
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_subjects)
    
    train_subjects = all_subjects[:N_TRAIN_SUBJECTS]
    val_subjects = all_subjects[N_TRAIN_SUBJECTS:N_TRAIN_SUBJECTS + N_VAL_SUBJECTS]
    test_subjects = all_subjects[N_TRAIN_SUBJECTS + N_VAL_SUBJECTS:]
    
    print(f"训练被试：{len(train_subjects)} 人")
    print(f"验证被试：{len(val_subjects)} 人")
    print(f"测试被试：{len(test_subjects)} 人")
    
    # 处理每个划分
    splits = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }
    
    all_data = {}
    
    for split_name, subjects in splits.items():
        print(f"\n{'='*60}")
        print(f"处理 {split_name} 集（{len(subjects)} 人）...")
        print(f"{'='*60}")
        
        split_windows = []
        split_labels = []
        
        for subj in subjects:
            # 静息态
            rest_file = INPUT_DIR / f"{subj}{REST_SUFFIX}"
            if rest_file.exists():
                print(f"  处理 {subj} 静息态...")
                w, l = process_one_edf(rest_file, label=0)
                split_windows.append(w)
                split_labels.append(l)
                print(f"    得到 {len(w)} 个窗口")
            else:
                print(f"  警告：{rest_file} 不存在，跳过")
            
            # 专注态
            focus_file = INPUT_DIR / f"{subj}{FOCUS_SUFFIX}"
            if focus_file.exists():
                print(f"  处理 {subj} 专注态...")
                w, l = process_one_edf(focus_file, label=1)
                split_windows.append(w)
                split_labels.append(l)
                print(f"    得到 {len(w)} 个窗口")
            else:
                print(f"  警告：{focus_file} 不存在，跳过")
        
        if len(split_windows) > 0:
            X = np.concatenate(split_windows, axis=0)
            y = np.concatenate(split_labels, axis=0)
        else:
            X = np.array([])
            y = np.array([])
        
        all_data[split_name] = (X, y)
        
        print(f"\n  {split_name} 集结果：")
        print(f"    X 形状：{X.shape}")
        print(f"    y 形状：{y.shape}")
        if len(y) > 0:
            print(f"    标签分布：静息={np.sum(y==0)}, 专注={np.sum(y==1)}")
            print(f"    比例：静息={np.sum(y==0)/len(y):.4f}, 专注={np.sum(y==1)/len(y):.4f}")
    
    # 计算训练集的class_weights
    if len(all_data["train"][1]) > 0:
        class_weights = compute_class_weights(all_data["train"][1], method="inverse")
    else:
        class_weights = [1.0, 1.0]
        print("警告：训练集为空，使用默认权重 [1.0, 1.0]")
    
    # 保存数据
    print("\n" + "=" * 60)
    print("保存数据...")
    print("=" * 60)
    
    for split_name in ["train", "val", "test"]:
        X, y = all_data[split_name]
        np.save(OUTPUT_DIR / f"X_{split_name}.npy", X)
        np.save(OUTPUT_DIR / f"y_{split_name}.npy", y)
        print(f"  已保存 X_{split_name}.npy, y_{split_name}.npy")
    
    # 保存元数据
    metadata = {
        "dataset_name": "mental_arithmetic",
        "description": "心算专注数据集（静息态 vs 专注态二分类）",
        "n_subjects_total": len(all_subjects),
        "n_subjects_train": N_TRAIN_SUBJECTS,
        "n_subjects_val": N_VAL_SUBJECTS,
        "n_subjects_test": N_TEST_SUBJECTS,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "sampling_rate_hz": TARGET_SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "n_channels": len(SELECTED_CHANNEL_NAMES),
        "channel_names": SELECTED_CHANNEL_NAMES,
        "channel_positions": CHANNEL_POSITIONS,
        "n_classes": 2,
        "class_names": ["rest (eyes open baseline)", "focus (mental arithmetic)"],
        "bandpass_filter": f"{BANDPASS_LOW}-{BANDPASS_HIGH} Hz",
        "class_weights_method": "inverse (简单反比)",
        "class_weights": class_weights,
        "random_seed": RANDOM_SEED,
        "train_samples": int(len(all_data["train"][1])),
        "val_samples": int(len(all_data["val"][1])),
        "test_samples": int(len(all_data["test"][1])),
        "train_class_distribution": {
            "rest": int(np.sum(all_data["train"][1] == 0)),
            "focus": int(np.sum(all_data["train"][1] == 1)),
        },
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n  已保存 metadata.json")
    
    # 最终总结
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"数据形状：(N, 8, 500)")
    print(f"采样率：{TARGET_SAMPLING_RATE}Hz")
    print(f"窗口：{WINDOW_SECONDS}秒（{WINDOW_SAMPLES}采样点）")
    print(f"分类：二分类（静息 vs 专注）")
    print(f"通道：{SELECTED_CHANNEL_NAMES}")
    print(f"训练集：{len(all_data['train'][1])} 样本")
    print(f"验证集：{len(all_data['val'][1])} 样本")
    print(f"测试集：{len(all_data['test'][1])} 样本")
    print(f"类别权重（方法A）：静息={class_weights[0]:.4f}, 专注={class_weights[1]:.4f}")
    print("=" * 60)
    print("\n下一步：")
    print("1. 用这个数据做AAMP预训练")
    print("2. 用class_weights做加权交叉熵监督微调")
    print("3. 看验证集混淆矩阵，调整class_weights")


if __name__ == "__main__":
    main()
