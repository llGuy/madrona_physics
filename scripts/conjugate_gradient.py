"""
Newton related solvers
"""
import numpy as np

from scripts.line_searches import my_line_search


def nonlinear_cg(fun, df, x0, tol, M):
    avg_tol = tol
    max_iter = len(x0) * 100

    # Initialize
    x = x0.copy()
    g = df(x)
    M_grad = M.solve(g)
    p = -M_grad

    for i in range(max_iter):
        # Convergence check
        if np.linalg.norm(g) < avg_tol:
            break

        # Compute step
        alpha = my_line_search(fun, df, x, p)
        if alpha is None:
            print("Line search failed")
            print("is descent direction?", np.dot(g, p) < 0)
            break

        # Next step
        x_new = x + alpha * p
        g_new = df(x_new)
        Mgrad_new = M.solve(g_new)

        # Polak-Ribiere
        beta = np.dot(g_new, (Mgrad_new - M_grad)) / np.dot(g, g)
        beta = max(0, beta)
        p_new = -Mgrad_new + beta * p

        # Update
        p, x, g, M_grad = p_new, x_new, g_new, Mgrad_new

    print("CG Iterations", i)
    return x
