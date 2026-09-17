"""
贝叶斯优化搜索算法 (Bayesian Optimization)

和准随机搜索的区别：
- 准随机搜索：每次试验独立，不利用之前的结果
- 贝叶斯优化：用之前的试验结果训练一个代理模型（高斯过程），
  然后用采集函数（EI/UCB/POI）选择下一个最有希望的试验点

贝叶斯优化在试验次数有限时通常比随机搜索更高效，
但实现更复杂，且高维空间效果会下降。

参考 AAD 项目的 bayesian_lr_search.py，这里重构为通用类。

支持的采集函数：
1. EI (Expected Improvement) - 推荐，平衡探索和利用
2. UCB (Upper Confidence Bound) - 更偏向探索
3. POI (Probability of Improvement) - 更偏向利用

Args:
    search_space: 搜索空间字典，键=参数名，值=[低,高,类型]
    n_trials: 总试验次数
    n_initial: 初始随机试验次数（用于训练初始代理模型）
    acquisition: 采集函数 "ei" / "ucb" / "poi"
    kernel: 高斯过程核 "matern" / "rbf"
    seed: 随机种子
    objective_metric: 目标指标名
    maximize: 是否最大化目标
    kappa: UCB 的探索参数（越大越探索）
    xi: EI/POI 的探索参数
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import json


class BayesianOptimization:
    """贝叶斯超参数优化"""

    def __init__(
        self,
        search_space: Dict[str, List],
        n_trials: int = 50,
        n_initial: int = 10,
        acquisition: str = "ei",
        kernel: str = "matern",
        seed: int = 42,
        objective_metric: str = "val_balanced_accuracy",
        maximize: bool = True,
        kappa: float = 2.576,
        xi: float = 0.01,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.kernel = kernel
        self.seed = seed
        self.objective_metric = objective_metric
        self.maximize = maximize
        self.kappa = kappa
        self.xi = xi

        self.param_names = list(search_space.keys())
        self.n_params = len(self.param_names)

        # 已观测的点和值
        self.observed_X: List[np.ndarray] = []  # 归一化空间的点
        self.observed_y: List[float] = []        # 目标值
        self.results: List[Dict] = []

        self.best_params: Optional[Dict] = None
        self.best_value: float = -float("inf") if maximize else float("inf")

        self.rng = np.random.RandomState(seed)
        self._gp_model = None

    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """把实际参数值编码为归一化空间 [0,1] 的点"""
        x = np.zeros(self.n_params)
        for i, name in enumerate(self.param_names):
            spec = self.search_space[name]
            if len(spec) == 4 and all(not isinstance(v, str) for v in spec):
                value = params[name]
                distances = np.abs(np.asarray(spec, dtype=float) - float(value))
                x[i] = 0.0 if distances.max() == 0 else float(distances.argmin()) / (len(spec) - 1)
                continue
            low, high, _ = spec
            value = params[name]
            x[i] = (value - low) / (high - low)
        return np.clip(x, 0.0, 1.0)

    def _decode_sample(self, sample: np.ndarray) -> Dict[str, Any]:
        """把归一化空间的点解码为实际参数值"""
        params = {}
        for i, name in enumerate(self.param_names):
            spec = self.search_space[name]
            if len(spec) == 4 and all(not isinstance(v, str) for v in spec):
                params[name] = spec[min(int(sample[i] * len(spec)), len(spec) - 1)]
                continue
            low, high, param_type = spec
            value = low + sample[i] * (high - low)
            if param_type == "int":
                value = int(round(value))
                value = max(low, min(high, value))
            elif param_type == "float":
                value = float(value)
            params[name] = value
        return params

    def _random_sample(self) -> np.ndarray:
        """随机采样一个点（用于初始阶段）"""
        return self.rng.rand(self.n_params)

    def _fit_gp(self):
        """训练高斯过程代理模型"""
        if len(self.observed_X) < 2:
            return

        X = np.array(self.observed_X)
        y = np.array(self.observed_y)

        # 标准化 y
        self._y_mean = y.mean()
        self._y_std = y.std() + 1e-8
        y_norm = (y - self._y_mean) / self._y_std

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel

            if self.kernel == "matern":
                kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=0.5)

            self._gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.seed,
            )
            self._gp_model.fit(X, y_norm)
        except ImportError:
            # 没有 sklearn，用简化的最近邻回归
            self._gp_model = None
            print("警告: 未安装 scikit-learn，贝叶斯优化退化为随机搜索")

    def _acquisition(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        计算采集函数值

        Args:
            X_candidates: [n_candidates, n_params]

        Returns:
            acquisition values: [n_candidates]
        """
        if self._gp_model is None:
            # 退化为随机
            return self.rng.rand(len(X_candidates))

        mu, sigma = self._gp_model.predict(X_candidates, return_std=True)
        sigma = sigma + 1e-8

        # 当前最佳（标准化空间）
        if self.maximize:
            best_y_norm = max(self.observed_y)
            best_y_norm = (best_y_norm - self._y_mean) / self._y_std
        else:
            best_y_norm = min(self.observed_y)
            best_y_norm = (best_y_norm - self._y_mean) / self._y_std

        if self.acquisition == "ei":
            # Expected Improvement
            if self.maximize:
                improvement = mu - best_y_norm - self.xi
            else:
                improvement = best_y_norm - mu - self.xi
            Z = improvement / sigma
            ei = improvement * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)
            return np.maximum(ei, 0)

        elif self.acquisition == "ucb":
            # Upper Confidence Bound
            if self.maximize:
                return mu + self.kappa * sigma
            else:
                return -(mu - self.kappa * sigma)

        elif self.acquisition == "poi":
            # Probability of Improvement
            if self.maximize:
                Z = (mu - best_y_norm - self.xi) / sigma
            else:
                Z = (best_y_norm - mu - self.xi) / sigma
            return self._norm_cdf(Z)

        else:
            return self.rng.rand(len(X_candidates))

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """标准正态分布 CDF"""
        from scipy.stats import norm
        return norm.cdf(x)

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """标准正态分布 PDF"""
        from scipy.stats import norm
        return norm.pdf(x)

    def _suggest_next(self) -> np.ndarray:
        """
        建议下一个试验点

        用采集函数在随机候选点中选择最优的。
        """
        # 生成大量随机候选点
        n_candidates = 10000
        candidates = self.rng.rand(n_candidates, self.n_params)

        # 计算采集函数值
        acq_values = self._acquisition(candidates)

        # 选择采集函数值最大的点
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]

    def get_next_params(self) -> Dict[str, Any]:
        """获取下一次试验的参数"""
        n_observed = len(self.observed_X)

        if n_observed < self.n_initial:
            # 初始阶段：随机采样
            sample = self._random_sample()
        else:
            # 贝叶斯优化阶段
            self._fit_gp()
            sample = self._suggest_next()

        return self._decode_sample(sample)

    def record_result(self, params: Dict[str, Any], metrics: Dict[str, float]):
        """记录一次试验的结果"""
        value = metrics.get(self.objective_metric, 0.0)

        # 记录归一化空间的点
        x = self._encode_params(params)
        self.observed_X.append(x)
        self.observed_y.append(value)

        result = {
            "trial_idx": len(self.results),
            "params": params,
            "metrics": metrics,
            "objective_value": value,
            "phase": "initial" if len(self.results) < self.n_initial else "bayesian",
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
        """运行完整的贝叶斯优化"""
        print(f"开始贝叶斯优化，采集函数={self.acquisition}, 核={self.kernel}")
        print(f"总试验次数={self.n_trials}, 初始随机={self.n_initial}")
        print(f"搜索空间: {self.search_space}")

        for i in range(self.n_trials):
            params = self.get_next_params()
            phase = "初始随机" if i < self.n_initial else "贝叶斯优化"
            print(f"\n试验 {i+1}/{self.n_trials} ({phase}): {params}")

            try:
                metrics = objective_fn(params)
                self.record_result(params, metrics)
                value = metrics.get(self.objective_metric, 0.0)
                print(f"  目标指标 {self.objective_metric}: {value:.4f}")
                print(f"  当前最佳: {self.best_value:.4f}")
            except Exception as e:
                print(f"  试验失败: {e}")
                self.record_result(params, {self.objective_metric: -float("inf") if self.maximize else float("inf")})

            if output_dir:
                self.save_results(output_dir)

        summary = self.get_summary()
        if output_dir:
            self.save_results(output_dir)

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """获取优化结果汇总"""
        if not self.results:
            return {"best_params": None, "best_value": None, "n_completed": 0}

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
            "n_initial": self.n_initial,
            "acquisition": self.acquisition,
            "kernel": self.kernel,
            "top_5": sorted_results[:5],
            "all_results": self.results,
        }

    def save_results(self, output_dir: str):
        """保存优化结果到 JSON"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        with open(Path(output_dir) / "bayesian_optimization_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
