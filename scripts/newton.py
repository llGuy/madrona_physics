"""
Newton related solvers
"""
import numpy as np
from scipy.sparse.linalg import spsolve

from line_search import exact_line_search


def newton(df, hess, x0, tol, M, a_free, J, a_ref, mus):
    """
    Minimizes [fun] using Newton's method.
    The search direction [hess]^{-1}[df] is computed using the conjugate gradient method.
    """
    x = x0.copy()
    avg_tol = tol * len(x0)
    max_iter = len(x0) * 100
    for i in range(max_iter):
        # Newton step
        g = df(x)
        if np.linalg.norm(g) < avg_tol:
            break
        H = hess(x)
        p = spsolve(H, -g)

        # Exact line search
        alpha = exact_line_search(x, p, tol, avg_tol, M, a_free, J, a_ref, mus)
        update = alpha * p
        x += update

        if np.linalg.norm(update) < avg_tol:
            break

    print("Newton iterations:", i)
    return x
