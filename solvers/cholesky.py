from random import triangular
from solvers.solver import Solver
import numpy as np
from scipy.linalg import solve_triangular

class Cholesky(Solver):
    def __init__(self):
        super(Cholesky, self).__init__()

    def fit(self, A, b):
        L = np.linalg.cholesky(A)

        # y = solve_triangular(L, b, lower=True)
        # x = solve_triangular(L.T, y, lower=False)
        
        y = self.triangular_solve(L, b)
        x = self.triangular_solve(L.T, y, lower=False)

        return x


    def triangular_solve(self, A, b, lower=True):
        '''
        lower: use lower triangular entries
        '''
        matrix_size = A.shape[0]
        x = np.zeros(matrix_size)
        for i in range(matrix_size):
            if lower: x[i] = (b[i] - (A[i, 0:i] @ x[0:i]))/A[i,i]
            else: 
                x[matrix_size-i-1] = (b[matrix_size-i-1] - (A[matrix_size-i-1][matrix_size-i:] @ x[matrix_size-i:]))/A[matrix_size-i-1,matrix_size-i-1]

        return x


