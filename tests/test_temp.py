import pandas as pd
import numpy as np

from pyticle.benchmark import Benchmark
from pyticle.meta_search import MetaSearch
from pyticle.swarm_optimization import SwarmOptimization


def get_problem_input():
    cost_func = Benchmark.ackley
    pn = int(np.random.randint(low=50, high=200, size=1))
    os = max(np.random.rand() - 0.5, 0.0)
    oe = min(np.random.rand() + 0.5, 1.0)
    cf = np.random.rand(2) + 1.5
    bs = np.random.choice(["random", "clipping"], p=[0.3, 0.7])
    mi = int(np.random.randint(low=100, high=500, size=1))
    er = np.random.rand() * 0.2
    low_bound = -1
    high_bound = 1
    var_size = 2
    return dict(
        cost_func=cost_func,
        particle_num=pn,
        omega_start=os,
        omega_end=oe,
        coef=cf,
        low_bound=low_bound,
        high_bound=high_bound,
        boundary_strategy=bs,
        var_size=var_size,
        max_iter_num=mi,
        elite_rate=er,
    )


def test_boundary_strategy_random():
    inp = get_problem_input()
    inp["boundary_strategy"] = "random"
    optimizer = SwarmOptimization(**inp)
    optimizer.optimize()


def test_boundary_strategy_clipping():
    inp = get_problem_input()
    inp["boundary_strategy"] = "clipping"
    optimizer = SwarmOptimization(**inp)
    optimizer.optimize()


def test_meta_search_execution():
    ms = MetaSearch(
        cost_func=Benchmark.ackley, var_size=2, low_bound=-32, high_bound=32
    )
    result = ms.search(try_num=3, n_jobs=1)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"meta_search results should be "
            f"pandas.DataFrame, and not {type(result)}"
        )
    if result.shape[0] != 3:
        raise ValueError(
            f"meta_search results should have 3 rows, and not {result.shape[0]}"
        )
    if result.shape[1] != 13:
        raise ValueError(
            f"meta_search results should have 13 columns, and not {result.shape[1]}"
        )
    if len(result.loc[0, "x_opt"]) != 2:
        raise ValueError(
            f'meta_search x* should be in 2D space, and not {len(result.loc[0, "x_opt"])}'
        )
