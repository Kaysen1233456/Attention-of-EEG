"""
准随机搜索算法 (Quasi-Random Search)

和普通随机搜索的区别：
- 普通随机搜索：每个点独立随机采样，可能出现聚集或空隙
- 准随机搜索：用低差异序列（Sobol/Halton/Latin Hypercube）采样，
  点在空间中分布更均匀，覆盖更全面，同样的试验次数能找到更好的解

参考 AAD 项目的 lr_bs_quasi_random_search.py，这里重构为通用类。

支持的序列：
1. Sobol 序列（推荐，高维性能好）
2. Halton 序列
3. Latin Hypercube（拉丁超立方）

Args:
    search_space: 搜索空间字典，键=参数名，值=[低,高,类型]
        类型: "float" 或 "int"
        示例: {"learning_rate": [1e-4, 5e-3, "float"], "batch_size": [16, 64, "int"]}
    n_trials: 试验次数
    method: "sobol" / "halton" / "latin_hypercube"
    seed: 随机种子
    objective_metric: 目标指标名（用于排序）
    maximize: 是否最大化目标（True=最大化，False=最小化）
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import json


class QuasiRandomSearch:
    """准随机超参数搜索"""

    def __init__(
        self,
        search_space: Dict[str, List],
        n_trials: int = 50,
        method: str = "sobol",
        seed: int = 42,
        objective_metric: str = "val_balanced_accuracy",
        maximize: bool = True,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.method = method
        self.seed = seed
        self.objective_metric = objective_metric
        self.maximize = maximize

        # 参数名列表（保持顺序）
        self.param_names = list(search_space.keys())
        self.n_params = len(self.param_names)

        # 生成采样点
        self.samples = self._generate_samples()

        # 结果记录
        self.results: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_value: float = -float("inf") if maximize else float("inf")

    def _generate_samples(self) -> np.ndarray:
        """
        生成准随机采样点（归一化到 [0,1] 空间）

        Returns:
            samples: [n_trials, n_params]，值在 [0,1] 范围内
        """
        rng = np.random.RandomState(self.seed)

        if self.method == "sobol":
            samples = self._sobol_sequence(self.n_trials, self.n_params, self.seed)
        elif self.method == "halton":
            samples = self._halton_sequence(self.n_trials, self.n_params)
        elif self.method == "latin_hypercube":
            samples = self._latin_hypercube(self.n_trials, self.n_params, rng)
        else:
            raise ValueError(f"未知的准随机方法: {self.method}")

        # 确保在 [0,1] 范围内
        samples = np.clip(samples, 0.0, 1.0)
        return samples

    @staticmethod
    def _sobol_sequence(n: int, d: int, seed: int = 42) -> np.ndarray:
        """
        生成 Sobol 序列

        优先使用 scipy.stats.qmc.Sobol，如果没有则用简化实现。
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
            return sampler.random(n)
        except ImportError:
            # 简化实现：用随机数代替（功能等价但分布稍差）
            rng = np.random.RandomState(seed)
            return rng.rand(n, d)

    @staticmethod
    def _halton_sequence(n: int, d: int) -> np.ndarray:
        """
        生成 Halton 序列

        用前 d 个质数作为底数。
        """
        def halton_1d(index: int, base: int) -> float:
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f /= base
            return result

        # 前 d 个质数
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        if d > len(primes):
            # 超过15维，用随机数
            rng = np.random.RandomState(42)
            return rng.rand(n, d)

        samples = np.zeros((n, d))
        for j in range(d):
            base = primes[j]
            for i in range(n):
                samples[i, j] = halton_1d(i + 1, base)
        return samples

    @staticmethod
    def _latin_hypercube(n: int, d: int, rng: np.random.RandomState) -> np.ndarray:
        """
        生成拉丁超立方采样

        每一行每一列都只有一个点，保证边缘分布均匀。
        """
        samples = np.zeros((n, d))
        for j in range(d):
            # 把 [0,1] 分成 n 个区间，每个区间随机取一个点
            perm = rng.permutation(n)
            for i in range(n):
                samples[i, j] = (perm[i] + rng.rand()) / n
        return samples

    def _decode_sample(self, sample: np.ndarray) -> Dict[str, Any]:
        """
        把归一化的采样点 [0,1] 解码为实际参数值

        Args:
            sample: [n_params]，值在 [0,1]

        Returns:
            参数字典
        """
        params = {}
        for i, name in enumerate(self.param_names):
            spec = self.search_space[name]
            if len(spec) == 4 and all(not isinstance(v, str) for v in spec):
                index = min(int(sample[i] * len(spec)), len(spec) - 1)
                params[name] = spec[index]
                continue
            low, high, param_type = spec
            if param_type == "log_float":
                value = math.exp(math.log(low) + sample[i] * (math.log(high) - math.log(low)))
            else:
                value = low + sample[i] * (high - low)

            if param_type == "int":
                value = int(round(value))
                value = max(low, min(high, value))  # 确保在范围内
            elif param_type in ("float", "log_float"):
                value = float(value)

            params[name] = value
        return params

    def get_trial_params(self, trial_idx: int) -> Dict[str, Any]:
        """获取第 trial_idx 次试验的参数"""
        if trial_idx >= self.n_trials:
            raise IndexError(f"试验索引 {trial_idx} 超出范围 (0-{self.n_trials-1})")
        return self._decode_sample(self.samples[trial_idx])

    def record_result(self, trial_idx: int, params: Dict[str, Any], metrics: Dict[str, float]):
        """
        记录一次试验的结果

        Args:
            trial_idx: 试验索引
            params: 参数字典
            metrics: 评估指标字典
        """
        if self.objective_metric not in metrics:
            raise ValueError(
                f"Missing objective metric '{self.objective_metric}' in {metrics}"
            )
        value = float(metrics[self.objective_metric])
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite objective metric '{self.objective_metric}': {value}"
            )

        result = {
            "trial_idx": trial_idx,
            "params": params,
            "metrics": metrics,
            "objective_value": value,
        }
        self.results.append(result)

        # 更新最佳
        if self.maximize:
            if value > self.best_value:
                self.best_value = value
                self.best_params = params
        else:
            if value < self.best_value:
                self.best_value = value
                self.best_params = params

    def run(
        self,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的准随机搜索

        Args:
            objective_fn: 目标函数，输入参数字典，返回评估指标字典
            output_dir: 结果保存目录（可选）

        Returns:
            搜索结果汇总
        """
        print(f"开始准随机搜索，方法={self.method}, 试验次数={self.n_trials}")
        print(f"搜索空间: {self.search_space}")

        for i in range(self.n_trials):
            params = self.get_trial_params(i)
            params["trial_idx"] = i
            print(f"\n试验 {i+1}/{self.n_trials}: {params}")

            try:
                metrics = objective_fn(params)
                self.record_result(i, params, metrics)
                value = metrics[self.objective_metric]
                print(f"  目标指标 {self.objective_metric}: {value:.4f}")
                print(f"  当前最佳: {self.best_value:.4f}")
            except Exception as e:
                print(f"  试验失败: {e}")
                self.results.append({
                    "trial_idx": i,
                    "params": params,
                    "metrics": {},
                    "objective_value": -float("inf") if self.maximize else float("inf"),
                    "status": "failed",
                    "error": repr(e),
                })

            # 保存中间结果
            if output_dir:
                self.save_results(output_dir)

        # 最终结果
        summary = self.get_summary()
        if output_dir:
            self.save_results(output_dir)

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """获取搜索结果汇总"""
        if not self.results:
            return {"best_params": None, "best_value": None, "n_completed": 0}

        # 按目标指标排序
        sorted_results = sorted(
            self.results,
            key=lambda x: x["objective_value"],
            reverse=self.maximize,
        )

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "objective_metric": self.objective_metric,
            "maximize": self.maximize,
            "n_completed": len(self.results),
            "n_trials": self.n_trials,
            "method": self.method,
            "top_5": sorted_results[:5],
            "all_results": self.results,
        }

    def save_results(self, output_dir: str):
        """保存搜索结果到 JSON"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        with open(Path(output_dir) / "quasi_random_search_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def get_optimal_lr_range(self, n_top: int = 5) -> Tuple[float, float]:
        """
        获取最优学习率区间（AAD 项目用过的分析方法）

        取目标指标最好的 n_top 个试验，返回它们的学习率范围。
        """
        if "learning_rate" not in self.param_names:
            return (0.0, 0.0)

        sorted_results = sorted(
            self.results,
            key=lambda x: x["objective_value"],
            reverse=self.maximize,
        )
        top_lrs = [r["params"]["learning_rate"] for r in sorted_results[:n_top]]
        return (min(top_lrs), max(top_lrs))
