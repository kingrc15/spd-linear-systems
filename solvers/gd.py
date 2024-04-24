from solvers.solver import Solver

class GD(Solver):
    def __init__(self):
        super(GD, self).__init__()

    def fit(self, A, b):
        raise NotImplementedError()
