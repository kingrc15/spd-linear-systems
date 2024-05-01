from typing import Optional
from solvers.solver import Solver
import numpy as np
from scipy.sparse import csr_matrix

class GD(Solver):
    def __init__(self, num_iterations:int, target_residual:float, lr:Optional[float]=None, use_sparse:bool=False):
        super(GD, self).__init__()
        self.lr = lr
        self.num_iterations = num_iterations
        self.target_residual_squared = target_residual * target_residual
        self.use_sparse = use_sparse
        self.lrs = []

    def fit(self, A, b):
        x = np.zeros_like(b)

        if self.use_sparse:
            A = csr_matrix(A)
            # b = csr_matrix(b)
            # self.x = csr_matrix(self.x)

        for _ in range(self.num_iterations):
            x, r = self.step(A, b, x)
            if r < self.target_residual_squared:
                break

        return x
    
    def step(self, A, b, x):
        p = A @ x - b
        if self.lr is None:
            lr = np.dot(p, p) / np.dot(p, A @ p)
        else:
            lr = self.lr
        self.lrs.append(lr)
        # we return the mean residual squared
        # not returning norm of residual since it is affected by n
        return x - lr * p, np.mean(p ** 2)  
