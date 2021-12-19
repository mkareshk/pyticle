from tabulate import tabulate

from pyticle.benchmark import Benchmark
from pyticle.meta_search import MetaSearch

ms = MetaSearch(cost_func=Benchmark.ackley, var_size=2, low_bound=-32, high_bound=32)
result = ms.search(try_num=8, n_jobs=8)
print(tabulate(result, headers='keys', tablefmt='psql'))
