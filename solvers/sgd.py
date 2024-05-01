import numpy as np
from solvers.solver import Solver

class SGD(Solver):
    def __init__(
        self,
        iterations=100,
        batch_size=1,  
        step_size=1e-3,
        iterate_mode="stochastic"
        ):
        super(SGD, self).__init__()
        self.iterations = iterations
        self.batch_size = batch_size
        self.step_size = step_size
        self.iterate_mode = iterate_mode

    def init_variables(self, A, b):
        self.A = A
        self.b = b
        self.start_idx = 0
        self.end_idx = min(self.batch_size, len(self.A))
        self.x = np.random.randn(len(self.A))
        self.loss = []
        self.compute_loss()

        if self.iterate_mode == "stochastic":
            self.indices = np.random.permutation(len(self.A))
        elif self.iterate_mode == "cyclical":
            self.indices = np.arange(len(self.A))
        else:
            raise NotImplementedError()

    def compute_loss(self):
        residuals = (self.A @ self.x - self.b) ** 2
        residual = residuals.mean()
        self.loss.append(residual)

    def update_batch(self):
        self.start_idx += self.batch_size
        self.end_idx += self.batch_size
        self.end_idx = min(len(self.A), self.end_idx)
        if self.start_idx >= len(self.A):
            if self.iterate_mode == "stochastic":
                np.random.shuffle(self.indices)
            self.start_idx = 0 
            self.end_idx = min(self.batch_size, len(self.A))

    def sgd_iterate(self):
        batch_indices = self.indices[self.start_idx:self.end_idx]
        self.A_hat = self.A[batch_indices]
        self.b_hat = self.b[batch_indices]
        self.compute_grad()
        self.update_solution()
        self.compute_loss()

    def compute_grad(self):
        temp = self.A_hat @ self.x - self.b_hat
        self.grad = self.A_hat.T @ temp / len(self.b_hat) #normalize the gradient

    def update_solution(self):
        self.x -= self.step_size * self.grad

    def fit(self, A, b, return_loss=False):
        self.init_variables(A, b)
        for _ in range(self.iterations):
            self.sgd_iterate()
            self.update_batch()
        if return_loss:
            return self.x, self.loss
        else:
            return self.x 
