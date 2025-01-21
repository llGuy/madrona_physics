"""
Preconditioned Nonlinear Conjugate Gradient solver with
    mass matrix M used as pre-conditioner
"""
import numpy as np

from line_search import exact_line_search


def nonlinear_cg(f, df, x0, tol, ls_tol, M, a_free, J, a_ref, mus, precision):
    max_iter = len(x0)
    # Scale the tolerance by the diagonal sum of the mass matrix
    # total_m = M.diag_sum()
    # scale = 1 / total_m
    scale = np.sqrt(1 / x0.size)

    # Initialize
    fun = f(x0)
    x = x0.copy()
    g = df(x)
    M_grad = M.solve(g)
    p = -M_grad

    for i in range(max_iter):
        # Convergence check
        if scale * np.linalg.norm(g) < tol:
            break

        # Exact line search
        # line_search_tol = tol * ls_tol * np.linalg.norm(p) / scale
        line_search_tol = ls_tol
        alpha = exact_line_search(x, p, line_search_tol, M, a_free, J, a_ref,
                                  mus, precision)
        if alpha == 0:
            break
        update = alpha * p

        # Polak-Ribiere
        x_new = x + update
        g_new = df(x_new)
        Mgrad_new = M.solve(g_new)
        fun_new = f(x_new)

        # Check improvement
        if scale * (fun - fun_new) < tol:
            break

        beta = np.dot(g_new, (Mgrad_new - M_grad)) / max(np.dot(g, M_grad), 1e-12)
        beta = max(0, beta)
        p_new = -Mgrad_new + beta * p
        x, g, p, M_grad, fun = x_new, g_new, p_new, Mgrad_new, fun_new

    print("CG Iterations", i)
    return x
