"""
Preconditioned Nonlinear Conjugate Gradient solver with
    mass matrix M used as pre-conditioner
"""
import numpy as np

from line_search import exact_line_search


def nonlinear_cg(df, x0, tol, M, a_free, J, a_ref, mus):
    max_iter = len(x0) * 100
    avg_tol = tol * len(x0)

    # Initialize
    x = x0.copy()
    g = df(x)
    M_grad = M.solve(g)
    p = -M_grad

    for i in range(max_iter):
        # Convergence check
        if np.linalg.norm(g) < avg_tol:
            break

        # Exact line search
        alpha = exact_line_search(x, p, tol, avg_tol, M, a_free, J, a_ref, mus)
        update = alpha * p

        # Polak-Ribiere
        x_new = x + update
        g_new = df(x_new)
        Mgrad_new = M.solve(g_new)

        beta = np.dot(g_new, (Mgrad_new - M_grad)) / np.dot(g, M_grad)
        beta = max(0, beta)
        p_new = -Mgrad_new + beta * p
        x, g, p, M_grad = x_new, g_new, p_new, Mgrad_new

        if np.linalg.norm(update) < avg_tol:
            break

    print("CG Iterations", i)
    return x
