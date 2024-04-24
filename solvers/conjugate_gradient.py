import numpy as np
from solvers.solver import Solver


class ConjugateGradient(Solver):
    def __init__(self, iterations):
        super(ConjugateGradient, self).__init__()
        self.iterations = iterations

    def fit(self, A, b):
        return np.random.randn(b.shape[0])
