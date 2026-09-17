"""
探查 Ear-SAAD .mat 文件结构
读取第一个被试的数据，查看变量名、维度、数据类型
"""
import scipy.io as sio
import numpy as np
from pathlib import Path

mat_path = Path(r"D:\Attention\data\raw\ear_saad\preprocessedData\preprocessedData\dataSubject1.mat")

print("=" * 70)
print(f"读取文件: {mat_path.name}")
print(f"文件大小: {mat_path.stat().st_size / 1024 / 1024:.1f} MB")
print("=" * 70)

# 读取mat文件
mat = sio.loadmat(mat_path)

# 过滤掉MATLAB内部变量
keys = [k for k in mat.keys() if not k.startswith('__')]
print(f"\n变量列表 ({len(keys)}个):")
for key in sorted(keys):
    val = mat[key]
    if isinstance(val, np.ndarray):
        print(f"  {key:30s} shape={str(val.shape):20s} dtype={val.dtype}")
    else:
        print(f"  {key:30s} type={type(val).__name__}")

# 详细查看每个变量
print("\n" + "=" * 70)
print("详细信息")
print("=" * 70)

for key in sorted(keys):
    val = mat[key]
    print(f"\n--- {key} ---")
    if isinstance(val, np.ndarray):
        print(f"  shape: {val.shape}")
        print(f"  dtype: {val.dtype}")
        if val.size > 0:
            if np.issubdtype(val.dtype, np.number):
                print(f"  min: {np.min(val):.4f}, max: {np.max(val):.4f}, mean: {np.mean(val):.4f}")
                print(f"  前5个值: {val.flatten()[:5]}")
            elif val.dtype.kind in ('U', 'S'):
                print(f"  字符串内容: {val}")
        # 如果是结构体数组
        if val.dtype.names:
            print(f"  结构体字段: {val.dtype.names}")
            for field in val.dtype.names:
                field_val = val[field]
                if isinstance(field_val, np.ndarray):
                    print(f"    {field}: shape={field_val.shape}, dtype={field_val.dtype}")
    elif isinstance(val, str):
        print(f"  字符串: {val}")
    else:
        print(f"  类型: {type(val)}")
        print(f"  值: {val}")

print("\n" + "=" * 70)
print("探查完成")
print("=" * 70)
