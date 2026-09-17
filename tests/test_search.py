"""
超参数搜索算法单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from attention_model.search.quasi_random import QuasiRandomSearch
from attention_model.search.bayesian_optimization import BayesianOptimization


class TestQuasiRandomSearch:
    """测试准随机搜索"""

    def test_init_default(self):
        """测试默认初始化"""
        search_space = {"lr": [1e-4, 1e-2, "float"], "bs": [16, 64, "int"]}
        searcher = QuasiRandomSearch(search_space, n_trials=10)
        assert searcher.n_trials == 10
        assert searcher.n_params == 2
        assert searcher.param_names == ["lr", "bs"]

    def test_sobol_samples_shape(self):
        """测试 Sobol 序列采样形状"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=20, method="sobol")
        assert searcher.samples.shape == (20, 1)
        assert np.all(searcher.samples >= 0) and np.all(searcher.samples <= 1)

    def test_halton_samples_shape(self):
        """测试 Halton 序列采样形状"""
        search_space = {"lr": [1e-4, 1e-2, "float"], "bs": [16, 64, "int"]}
        searcher = QuasiRandomSearch(search_space, n_trials=20, method="halton")
        assert searcher.samples.shape == (20, 2)

    def test_latin_hypercube_samples(self):
        """测试拉丁超立方采样"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=20, method="latin_hypercube")
        assert searcher.samples.shape == (20, 1)

    def test_decode_float_param(self):
        """测试浮点参数解码"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=5)
        params = searcher._decode_sample(np.array([0.5]))
        assert 1e-4 <= params["lr"] <= 1e-2
        assert isinstance(params["lr"], float)

    def test_decode_int_param(self):
        """测试整数参数解码"""
        search_space = {"bs": [16, 64, "int"]}
        searcher = QuasiRandomSearch(search_space, n_trials=5)
        params = searcher._decode_sample(np.array([0.5]))
        assert 16 <= params["bs"] <= 64
        assert isinstance(params["bs"], int)

    def test_get_trial_params(self):
        """测试获取试验参数"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=5)
        params = searcher.get_trial_params(0)
        assert "lr" in params

    def test_record_result(self):
        """测试记录结果"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=5, objective_metric="acc", maximize=True)
        params = {"lr": 0.001}
        metrics = {"acc": 0.85, "loss": 0.3}
        searcher.record_result(0, params, metrics)
        assert len(searcher.results) == 1
        assert searcher.best_value == 0.85
        assert searcher.best_params == params

    def test_record_result_minimize(self):
        """测试最小化目标"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=5, objective_metric="loss", maximize=False)
        searcher.record_result(0, {"lr": 0.001}, {"loss": 0.5})
        searcher.record_result(1, {"lr": 0.002}, {"loss": 0.3})
        assert searcher.best_value == 0.3

    def test_run_with_simple_objective(self):
        """测试运行简单目标函数"""
        search_space = {"x": [0.0, 10.0, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=5, objective_metric="value", maximize=True)

        def objective(params):
            return {"value": - (params["x"] - 5) ** 2 + 25}

        summary = searcher.run(objective)
        assert summary["n_completed"] == 5
        assert summary["best_value"] is not None

    def test_get_optimal_lr_range(self):
        """测试获取最优学习率区间"""
        search_space = {"learning_rate": [1e-4, 1e-2, "float"]}
        searcher = QuasiRandomSearch(search_space, n_trials=10, objective_metric="acc", maximize=True)
        for i in range(10):
            params = searcher.get_trial_params(i)
            searcher.record_result(i, params, {"acc": np.random.rand()})
        lr_range = searcher.get_optimal_lr_range(n_top=3)
        assert lr_range[0] <= lr_range[1]


class TestBayesianOptimization:
    """测试贝叶斯优化"""

    def test_init_default(self):
        """测试默认初始化"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        opt = BayesianOptimization(search_space, n_trials=20, n_initial=5)
        assert opt.n_trials == 20
        assert opt.n_initial == 5
        assert opt.acquisition == "ei"

    def test_initial_phase_random(self):
        """测试初始阶段随机采样"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        opt = BayesianOptimization(search_space, n_trials=20, n_initial=5)
        for i in range(5):
            params = opt.get_next_params()
            opt.record_result(params, {"acc": np.random.rand()})
        # 前5个应该都是初始随机阶段
        assert all(r["phase"] == "initial" for r in opt.results[:5])

    def test_encode_decode_roundtrip(self):
        """测试编码解码往返"""
        search_space = {"lr": [1e-4, 1e-2, "float"], "bs": [16, 64, "int"]}
        opt = BayesianOptimization(search_space, n_trials=10)
        params = {"lr": 0.005, "bs": 32}
        x = opt._encode_params(params)
        decoded = opt._decode_sample(x)
        assert abs(decoded["lr"] - 0.005) < 1e-6
        assert decoded["bs"] == 32

    def test_record_result(self):
        """测试记录结果"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        opt = BayesianOptimization(search_space, n_trials=10, objective_metric="acc", maximize=True)
        params = {"lr": 0.001}
        opt.record_result(params, {"acc": 0.9})
        assert len(opt.results) == 1
        assert opt.best_value == 0.9

    def test_get_summary(self):
        """测试获取汇总"""
        search_space = {"lr": [1e-4, 1e-2, "float"]}
        opt = BayesianOptimization(search_space, n_trials=10)
        for i in range(3):
            params = opt.get_next_params()
            opt.record_result(params, {"acc": np.random.rand()})
        summary = opt.get_summary()
        assert summary["n_completed"] == 3
        assert "best_params" in summary
        assert "top_5" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
