#!/usr/bin/env python3
"""
贝叶斯优化搜索脚本（基于已有搜索结果继续深挖）

核心设计：直接复用项目里 search_hyperparams.py 的 make_objective_fn，
100% 沿用已验证的训练逻辑，不自己猜 API。

功能：
1. 加载准随机搜索（Sobol）结果作为贝叶斯优化的初始观测值
2. 用高斯过程 + EI 采集函数继续搜索
3. 支持断点续跑
4. 每完成一个trial自动保存进度

用法：
    PYTHONPATH=src python scripts/bayesian_search_continue.py \
        --initial-results artifacts/route_C_search_sobol/quasi_random_search_results.json \
        --config configs/route_C_search.yaml \
        --output artifacts/route_C_search_bayesian \
        --n-trials 30 \
        --search-epochs 30 \
        --seed 42

依赖：scikit-optimize (skopt)
    pip install scikit-optimize
"""
import sys
import argparse
import json
import time
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from attention_model.config import AttentionConfig
from attention_model.data import EEGDataset
from attention_model.training import set_seed
from scripts.search_hyperparams import make_objective_fn


def parse_args():
    parser = argparse.ArgumentParser(description="贝叶斯优化搜索（基于已有结果继续）")
    parser.add_argument("--initial-results", type=str, required=True,
                        help="已有搜索结果JSON文件路径（准随机搜索的结果）")
    parser.add_argument("--config", type=str, required=True,
                        help="基础配置文件（含search.search_space定义）")
    parser.add_argument("--output", type=str, default="artifacts/bayesian_search_continue")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="贝叶斯优化新增的trial数")
    parser.add_argument("--search-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acquisition", type=str, default="ei",
                        choices=["ei", "ucb", "pi"])
    parser.add_argument("--kappa", type=float, default=1.96)
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--data", type=str, default=None)
    return parser.parse_args()


def build_search_space(config):
    """从配置文件构建skopt搜索空间"""
    from skopt.space import Real, Integer

    ss = config.search.search_space
    dimensions = []
    param_names = []

    for name, spec in ss.items():
        low, high, ptype = spec[0], spec[1], spec[2]
        param_names.append(name)
        if ptype == "log_float":
            dimensions.append(Real(low, high, prior="log-uniform", name=name))
        elif ptype == "float":
            dimensions.append(Real(low, high, prior="uniform", name=name))
        elif ptype == "int":
            dimensions.append(Integer(low, high, name=name))
        else:
            raise ValueError(f"不支持的参数类型: {ptype}")

    return dimensions, param_names


def load_initial_results(results_path, param_names, default_cw=None):
    """加载已有搜索结果作为初始观测值

    对于准随机结果中缺失的 rest_weight/focus_weight，用 default_cw 填充。
    """
    with open(results_path) as f:
        data = json.load(f)

    if default_cw is None:
        default_cw = [0.8, 2.0]

    all_results = data.get("all_results", data.get("results", []))
    X_init = []
    y_init = []
    loaded = 0
    skipped = 0

    for r in all_results:
        params = r.get("params", {})
        metrics = r.get("metrics", {})
        bal_acc = metrics.get("balanced_accuracy", None)
        if bal_acc is None:
            skipped += 1
            continue

        x = []
        valid = True
        for name in param_names:
            if name in params:
                x.append(params[name])
            elif name == "rest_weight":
                x.append(float(default_cw[0]))
            elif name == "focus_weight":
                x.append(float(default_cw[1]))
            else:
                valid = False
                break
        if not valid:
            skipped += 1
            continue

        X_init.append(x)
        y_init.append(-bal_acc)  # skopt最小化，取负
        loaded += 1

    print(f"从 {results_path} 加载了 {loaded} 个初始观测值（跳过 {skipped} 个）")
    return X_init, y_init


def params_to_dict(x, param_names):
    """将skopt返回的参数列表转为dict"""
    params = {}
    for name, val in zip(param_names, x):
        if isinstance(val, (np.integer,)):
            params[name] = int(val)
        elif isinstance(val, (np.floating,)):
            params[name] = float(val)
        else:
            params[name] = val
    return params


def save_progress(output_dir, X, y, param_names, best_value, best_params, trial_counter):
    progress = {
        "param_names": param_names,
        "n_completed": len(X),
        "best_value": float(best_value),
        "best_params": best_params,
        "all_X": [[float(v) for v in x] for x in X],
        "all_y": [float(v) for v in y],
        "trial_counter": trial_counter,
    }
    with open(Path(output_dir) / "bayesian_progress.json", "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_progress(output_dir):
    progress_path = Path(output_dir) / "bayesian_progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return None


def main():
    args = parse_args()

    # 检查skopt
    try:
        from skopt import Optimizer
        from skopt.learning import GaussianProcessRegressor
        from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
    except ImportError:
        print("错误：需要安装 scikit-optimize")
        print("运行: pip install scikit-optimize")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 加载配置（和 search_hyperparams.py 完全一致） ===
    base_config_obj = AttentionConfig.from_yaml(args.config)
    if args.data:
        base_config_obj.data.data_dir = args.data
    if args.num_workers is not None:
        base_config_obj.training.num_workers = args.num_workers
    base_config_obj.output.output_dir = str(output_dir)

    base_config = {
        "_config_path": args.config,
        "output_dir": str(output_dir),
        "data_dir": base_config_obj.data.data_dir,
        "num_workers": base_config_obj.training.num_workers,
        "seed": args.seed,
    }

    # === 构建搜索空间 ===
    dimensions, param_names = build_search_space(base_config_obj)
    print(f"搜索空间 ({len(param_names)} 参数): {param_names}")

    # === 加载数据（和 search_hyperparams.py 完全一致） ===
    dataset = EEGDataset(
        base_config_obj.data.data_dir,
        normalize=base_config_obj.data.normalize,
        clip_std=base_config_obj.data.normalize_clip_std,
        include_test=False,
        normalization_mode=base_config_obj.data.normalization_mode,
        zero_channel_policy=base_config_obj.data.zero_channel_policy,
    )
    datasets = {
        "train": dataset.train_dataset,
        "val": dataset.val_dataset,
    }
    print(f"数据加载完成: train={len(datasets['train'])}, val={len(datasets['val'])}")

    # === 创建目标函数（直接复用项目里的 make_objective_fn） ===
    objective_fn = make_objective_fn(base_config, datasets, device, args.search_epochs)

    # === 加载断点进度或初始结果 ===
    progress = load_progress(args.output)
    if progress and progress.get("all_X"):
        print(f"发现断点进度，已完成 {progress['n_completed']} 个trial")
        X_all = [x for x in progress["all_X"]]
        y_all = [v for v in progress["all_y"]]
        trial_counter = progress.get("trial_counter", len(X_all))
    else:
        default_cw = base_config_obj.training.class_weights
        X_init, y_init = load_initial_results(args.initial_results, param_names, default_cw)
        X_all = X_init
        y_all = y_init
        trial_counter = len(X_init)

    # === 初始化贝叶斯优化器 ===
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=args.seed,
    )

    acq_map = {"ei": "EI", "ucb": "LCB", "pi": "PI"}
    acq_func = acq_map.get(args.acquisition.lower(), "EI")
    acq_func_kwargs = {}
    if acq_func == "LCB":
        acq_func_kwargs["kappa"] = args.kappa
    else:
        acq_func_kwargs["xi"] = args.xi

    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator=gpr,
        n_initial_points=0,
        acq_func=acq_func,
        acq_func_kwargs=acq_func_kwargs,
        random_state=args.seed,
    )

    # 把已有观测值tell给优化器
    if X_all:
        optimizer.tell(X_all, y_all)
        best_idx = int(np.argmin(y_all))
        best_value = -y_all[best_idx]
        best_params = params_to_dict(X_all[best_idx], param_names)
        print(f"当前最佳: BalAcc={best_value:.4f}")
        print(f"最佳参数: {best_params}")
    else:
        best_value = 0
        best_params = None

    # === 贝叶斯优化循环 ===
    print(f"\n开始贝叶斯优化，新增 {args.n_trials} 个trial")
    print(f"采集函数: {acq_func}, 探索参数: {acq_func_kwargs}")
    print("=" * 70)

    for i in range(args.n_trials):
        trial_idx = trial_counter + i
        print(f"\n--- Trial {trial_idx} (贝叶斯第{i+1}/{args.n_trials}) ---")

        # 推荐下一组参数
        x_next = optimizer.ask()
        params = params_to_dict(x_next, param_names)
        print(f"推荐参数: lr={params.get('learning_rate', 0):.6f}, "
              f"dropout={params.get('dropout', 0):.4f}, "
              f"wd={params.get('weight_decay', 0):.2e}, "
              f"bs={params.get('batch_size', 0)}, "
              f"cw=[{params.get('rest_weight', 0):.3f}, {params.get('focus_weight', 0):.3f}]")

        # 运行trial（直接调用项目里的objective_fn，100%复用已有训练逻辑）
        t0 = time.time()
        try:
            params_for_call = params.copy()
            params_for_call["trial_idx"] = trial_idx
            val_metrics = objective_fn(params_for_call)
            bal_acc = val_metrics.get("balanced_accuracy", 0.5)
        except Exception as e:
            print(f"  Trial失败: {e}")
            traceback.print_exc()
            bal_acc = 0.5

        elapsed = time.time() - t0
        print(f"  结果: BalAcc={bal_acc:.4f}, 耗时={elapsed:.1f}s")

        # 更新优化器
        optimizer.tell(x_next, -bal_acc)
        X_all.append(x_next)
        y_all.append(-bal_acc)

        # 更新最佳
        if bal_acc > best_value:
            best_value = bal_acc
            best_params = params.copy()
            print(f"  ★ 新最佳! BalAcc={best_value:.4f}")

        # 保存进度
        save_progress(
            output_dir=str(output_dir),
            X=X_all,
            y=y_all,
            param_names=param_names,
            best_value=best_value,
            best_params=best_params,
            trial_counter=trial_counter + i + 1,
        )

    # === 最终结果 ===
    print("\n" + "=" * 70)
    print("贝叶斯优化完成!")
    print(f"总trial数: {len(X_all)} (初始{trial_counter} + 新增{args.n_trials})")
    print(f"最佳BalAcc: {best_value:.4f}")
    print(f"最佳参数: {json.dumps(best_params, indent=2, ensure_ascii=False)}")

    final_results = {
        "best_params": best_params,
        "best_value": float(best_value),
        "n_total": len(X_all),
        "n_initial": trial_counter,
        "n_bayesian": args.n_trials,
        "param_names": param_names,
        "all_results": [
            {
                "trial_idx": i,
                "params": params_to_dict(X_all[i], param_names),
                "balanced_accuracy": float(-y_all[i]),
            }
            for i in range(len(X_all))
        ],
    }
    with open(output_dir / "bayesian_search_results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_dir / 'bayesian_search_results.json'}")


if __name__ == "__main__":
    main()
