from solvers.solver import Solver

class Cholesky(Solver):
    def __init__(self):
        super(Cholesky, self).__init__()

    def fit(self, A, b):
        raise NotImplementedError()
