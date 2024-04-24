from solvers.solver import Solver

class SGD(Solver):
    def __init__(self):
        super(SGD, self).__init__()

    def fit(self, A, b):
        raise NotImplementedError()
