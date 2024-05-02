import argparse
import time
from itertools import product
import numpy as np
import pandas as pd

from solvers import Cholesky, ConjugateGradient, GD, SGD


def parser():
    parser = argparse.ArgumentParser(description="SPD-Linear-Systems")
    parser.add_argument(
        "--solver",
        choices=["cholesky", "conjugate", "gd", "sgd"],
        default="gd",
    )
    parser.add_argument(
        "--ns",
        nargs="+",
        default=[
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        ],
        type=int,
        help="size of matrix",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        default=[
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            # 0.1,
            # 0.2,
        ],
        type=float,
        help="sparsity parameter",
    )
    parser.add_argument(
        "--step_size",
        nargs="+",
        default=0.001,
        type=float,
        help="step size for gradient-based methods",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size for stochastic gradient descent",
    )
    parser.add_argument("--sgd_iter_mode", 
        default="stochastic", 
        choices=["stochastic", "cyclical"], 
        type=str, 
        help="batch selection policy"
        )
    parser.add_argument(
        "--iterations", default=10000, type=int, help="number of update iterations"
    )
    parser.add_argument("--eps", default=1e-5, type=float, help="target mean residual")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()

    seeds = list(range(50))

    time_means = {}
    time_stds = {}

    result_means = {}
    result_stds = {}

    lrs_means = {}
    lrs_stds = {}

    for n, delta in product(args.ns, args.deltas):
        t_deltas = []
        residuals = []
        lrs = []
        for seed in seeds:
            np.random.seed(seed)

            A = np.triu(np.random.rand(n, n) * 2 - 1)
            A += A.T
            np.fill_diagonal(A, 1)

            assert (np.abs(A) <= 1).all()
            assert (A == A.T).all()
            assert (np.diagonal(A) == 1).all()
            # assert np.all(np.linalg.eigvals(A) >= 0), np.linalg.eigvals(A)

            mask = np.abs(A) > delta
            np.fill_diagonal(mask, False)

            assert (mask == mask.T).all()
            assert (~np.diagonal(mask)).all()
            assert (np.abs(A) <= 1).all()
            assert (A == A.T).all()
            assert (np.diagonal(A) == 1).all()

            A[mask] = 0.0

            b = np.random.randn(n)

            if args.solver == "cholesky":
                solver = Cholesky()
            elif args.solver == "conjugate":
                solver = ConjugateGradient(n, args.eps)
            elif args.solver == "gd":
                solver = GD(n, args.eps)
            elif args.solver == "sgd":
                solver = SGD(
                    iterations=args.iterations,
                    batch_size=args.batch_size,
                    step_size=args.step_size,
                    iterate_mode=args.sgd_iter_mode
                    )

            time_begin = time.time()
            x = solver.fit(A, b)
            time_end = time.time()

            residual = ((A @ x - b) ** 2).mean()

            header = "*" * 20
            header += f" n = {n}"
            header += f", iters = {args.iterations}"
            header += f", delta = {delta} "
            header += "*" * 20

            t_deltas.append(time_end - time_begin)
            residuals.append(residual)

            if args.solver == "gd":
                lrs = solver.lrs

        result_means[(n, delta)] = np.mean(residuals)
        result_stds[(n, delta)] = np.std(residuals)
        time_means[(n, delta)] = np.mean(t_deltas)
        time_stds[(n, delta)] = np.std(t_deltas)
        
        if args.solver == "gd":
            lrs_means[(n, delta)] = np.mean(lrs)
            lrs_stds[(n, delta)] = np.std(lrs)
        print(header)
        print(
            f"Time elapsed: Mean = {time_means[(n, delta)]}, STD = {time_stds[(n, delta)]}"
        )
        print(
            f"Residual: Mean = {result_means[(n, delta)]}, STD = {result_stds[(n, delta)]}"
        )

    indices = np.array(list(result_means.keys())).T.tolist()
    multi_index = pd.MultiIndex.from_arrays(indices, names=["N", "Delta"])
    time_means = pd.Series(time_means.values(), index=multi_index, name="Mean Run Time")
    time_stds = pd.Series(time_stds.values(), index=multi_index, name="Run Time STD")

    result_means = pd.Series(
        result_means.values(), index=multi_index, name="Mean Residuals"
    )
    result_stds = pd.Series(
        result_stds.values(), index=multi_index, name="Residuals STD"
    )

    if args.solver == "gd":
        lrs_means = pd.Series(lrs_means.values(), index=multi_index, name="Mean LR")
        lrs_stds = pd.Series(lrs_stds.values(), index=multi_index, name="LR STD")
    
    print(time_means)
    print(time_stds)

    print(result_means)
    print(result_stds)

    for matrix_size in args.ns:
        tm_plot = time_means[matrix_size].plot(
            legend=True,
            xlabel="Delta",
            ylabel="time(s)",
            label=f"Matrix Size = {matrix_size}",
            use_index=True,
            yerr=time_stds[matrix_size],
        )

    tm_plot.set_title("Mean Runtime")
    tm_plot.figure.savefig("Mean Run Time.png", bbox_inches="tight", dpi=100)
    tm_plot.clear()

    for matrix_size in args.ns:
        rm_plot = result_means[matrix_size].plot(
            legend=True,
            xlabel="Delta",
            ylabel="Residuals",
            label=f"Matrix Size = {matrix_size}",
            use_index=True,
            yerr=result_stds[matrix_size],
        )

    rm_plot.set_title("Mean Residuals")
    rm_plot.figure.savefig("Mean Residuals.png", bbox_inches="tight", dpi=500)
    rm_plot.clear()

    if args.solver == "gd":
        for matrix_size in args.ns:
            rm_plot = lrs_means[matrix_size].plot(
                legend=True,
                xlabel="Delta",
                ylabel="MeanLearning Rate",
                label=f"Matrix Size = {matrix_size}",
                use_index=True,
                yerr=lrs_stds[matrix_size],
            )

        rm_plot.set_title("Mean Learning Rate")
        rm_plot.figure.savefig("Mean lr.png", bbox_inches="tight", dpi=500)
        rm_plot.clear()
