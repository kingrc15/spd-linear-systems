import argparse
import time
import numpy as np

from solvers import Cholesky, ConjugateGradient, GD, SGD


def parser():
    parser = argparse.ArgumentParser(description="SPD-Linear-Systems")
    parser.add_argument(
        "--solver",
        choices=["cholesky", "conjugate", "gd", "sgd"],
        default="conjugate",
    )

    parser.add_argument("--n", default=200, type=int, help="size of matrix")
    parser.add_argument("--delta", default=0.01, type=float, help="sparsity parameter")
    parser.add_argument(
        "--iterations", default=10, type=int, help="number of update iterations"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()

    A = np.triu(np.random.randn(args.n, args.n))
    A += A.T
    np.fill_diagonal(A, 1)

    assert (A == A.T).all()
    assert (np.diagonal(A) == 1).all()

    mask = np.abs(np.triu(np.random.randn(args.n, args.n))) > args.delta

    mask |= mask.T
    np.fill_diagonal(mask, False)

    assert (mask == mask.T).all()
    assert (~np.diagonal(mask)).all()

    A[mask] = 0

    b = np.random.randn(args.n)

    if args.solver == "cholesky":
        solver = Cholesky()
    elif args.solver == "conjugate":
        solver = ConjugateGradient(args.iterations)
    elif args.solver == "gd":
        solver = GD()
    elif args.solver == "sgd":
        solver = SGD()

    time_begin = time.time()
    x = solver.fit(A, b)
    time_end = time.time()

    print(A.shape, x.shape)
    residual = ((A @ x - b) ** 2).sum()

    print(f"Time elapsed = {time_end - time_begin}")
    print(f"Residual = {residual}")
