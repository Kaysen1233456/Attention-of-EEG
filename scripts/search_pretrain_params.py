
"""
预训练超参数搜索脚本（准随机搜索Sobol）
使用 configs/pretrain_aamp_mental_arithmetic.yaml 作为基础配置
"""
import sys
import argparse
import subprocess
import json
import os
import re
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import math

def sobol_sequence(n, d, seed=42):
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        return sampler.random(n)
    except ImportError:
        rng = np.random.RandomState(seed)
        return rng.rand(n, d)

def decode_sample(sample, search_space):
    params = {}
    for i, (name, spec) in enumerate(search_space.items()):
        low, high, param_type = spec
        if param_type == "log_float":
            value = math.exp(math.log(low) + sample[i] * (math.log(high) - math.log(low)))
        else:
            value = low + sample[i] * (high - low)
        if param_type == "int":
            value = int(round(value))
            value = max(low, min(high, value))
        params[name] = value
    return params

def run_pretrial(params, config_path, output_dir, epochs=50):
    trial_dir = output_dir / f"trial_{params['trial_idx']}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "scripts/pretrain_aamp.py",
        "--config", config_path,
        "--output", str(trial_dir),
        "--epochs", str(epochs),
        "--batch-size", str(params["batch_size"]),
        "--lr", str(params["learning_rate"]),
        "--unlabeled-splits", "train_only",
        "--checkpoint-selection", "validation",
        "--seed", "42",
        "--num-workers", "4",
        "--device", "cuda",
    ]
    
    print(f"  运行: lr={params['learning_rate']:.6f}, bs={params['batch_size']}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    with open(trial_dir / "run.log", "w") as f:
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)
    
    val_loss = None
    for line in result.stdout.split("\n"):
        if "val_loss" in line.lower():
            numbers = re.findall(r"[\d.]+", line)
            if numbers:
                val_loss = float(numbers[-1])
    
    if val_loss is None:
        result_file = trial_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            val_loss = data.get("val_loss") or data.get("best_val_loss")
    
    if val_loss is None:
        metadata_file = trial_dir / "pretrain_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                data = json.load(f)
            val_loss = data.get("best_val_loss") or data.get("val_loss")
    
    return val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--config", type=str, default="configs/pretrain_aamp_mental_arithmetic.yaml")
    parser.add_argument("--output", type=str, default="artifacts/pretrain_search")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    search_space = {
        "learning_rate": [1e-5, 1e-3, "log_float"],
        "batch_size": [16, 128, "int"],
    }
    
    samples = sobol_sequence(args.n_trials, len(search_space), args.seed)
    
    results = []
    best_val_loss = float("inf")
    best_params = None
    
    print(f"开始预训练超参数搜索（Sobol准随机）")
    print(f"基础配置: {args.config}")
    print(f"搜索空间: {search_space}")
    print(f"试验次数: {args.n_trials}")
    
    for i in range(args.n_trials):
        params = decode_sample(samples[i], search_space)
        params["trial_idx"] = i
        
        print(f"\n试验 {i+1}/{args.n_trials}: lr={params['learning_rate']:.6f}, bs={params['batch_size']}")
        
        try:
            val_loss = run_pretrial(params, args.config, output_dir, args.epochs)
            
            if val_loss is not None:
                print(f"  val_loss: {val_loss:.4f}")
                results.append({
                    "trial_idx": i,
                    "params": params,
                    "val_loss": val_loss,
                    "status": "success",
                })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params
                    print(f"  ★ 新最佳! val_loss={val_loss:.4f}")
            else:
                print(f"  未找到val_loss，检查run.log")
                results.append({
                    "trial_idx": i,
                    "params": params,
                    "val_loss": None,
                    "status": "no_val_loss",
                })
        except Exception as e:
            print(f"  失败: {e}")
            results.append({
                "trial_idx": i,
                "params": params,
                "val_loss": None,
                "status": "failed",
                "error": str(e),
            })
        
        with open(output_dir / "search_results.json", "w") as f:
            json.dump({
                "best_params": best_params,
                "best_val_loss": best_val_loss,
                "all_results": results,
            }, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("搜索完成！")
    print(f"最佳 val_loss: {best_val_loss:.4f}")
    print(f"最佳参数: {best_params}")
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
