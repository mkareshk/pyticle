import numpy as np


class Particle:
    def __init__(self, low_bound: float, high_bound: float, var_size: int):
        """
        implements the Particle class
        :param low_bound: the lower bound of variables in the optimization problem
        :param high_bound: the higher bound of variables in the optimization problem
        :param var_size: the problem's dimension
        """

        self.low_bound = low_bound
        self.high_bound = high_bound
        self.var_size = var_size

        self.position = np.random.uniform(low=low_bound, high=high_bound, size=var_size)
        self.best_position = self.position
        self.best_cost = np.inf
        range_bound = high_bound - low_bound
        self.velocity = np.random.uniform(
            low=-range_bound, high=range_bound, size=var_size
        )
        self.cost = np.inf
