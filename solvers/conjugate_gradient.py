import numpy as np
from solvers.solver import Solver


class ConjugateGradient:
    def __init__(self, iterations, eps):
        super(ConjugateGradient, self).__init__()
        self.iterations = iterations
        self.eps = eps

    def fit(self, A, b):
        x = np.zeros(b.shape[0])
        r = b.copy()
        p = r.copy()

        for _ in range(self.iterations):
            # Step length
            r_dot = r.T @ r
            ap = A @ p
            alpha = r_dot / (p.T @ ap)

            # Next iterate
            x += alpha * p

            # Residual
            r -= alpha * ap

            if np.abs(r).mean() < 0.001:
                break

            # Update for direction
            beta = (r.T @ r) / r_dot

            # Search Direction
            p = r + beta * p

        return x
