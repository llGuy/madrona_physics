"""
Newton related solvers
"""
import numpy as np
from scipy.optimize import line_search
from scipy.sparse.linalg import spsolve


def inner_newton_cg(z0, g, r, d, H, tol, cg_max_iter):
    """
    Currently unused (but may be helpful for larger systems)
    Inner loop for Newton-CG, solves for the search direction p in the linear system
        Hp = -g, where H is the Hessian and g is the gradient.
    """
    z = z0.copy()
    float64eps = np.finfo(np.float64).eps

    rs_old = np.dot(r, r)
    for j in range(cg_max_iter):
        # Check for convergence
        if np.linalg.norm(r) < tol:
            return z, 0

        # Curvature is small
        dBd = np.dot(d, (H.dot(d)))
        if 0 <= dBd <= 3 * float64eps:
            return z, 0
        # z is both a descent direction and a direction of non-positive curvature
        elif dBd <= 0:
            if j == 0:
                return -g, 0
            else:
                return z, 0

        # Continue iterating
        alpha = rs_old / dBd
        r += alpha * (H.dot(d))
        z += alpha * d

        rs_new = np.dot(r, r)
        beta = rs_new / rs_old
        d = -r + beta * d

        rs_old = rs_new
    else:
        return z0, 1



def newton(fun, df, hess, x0, tol):
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
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        p = spsolve(H, -g)

        # Line search
        alpha = line_search(fun, df, x, p, g)[0]
        if alpha is None:
            print("Is descent direction?", np.dot(g, p) < 0)
            print("Line search failed")  # why :(
            return x

        update = alpha * p
        x += update

        if np.linalg.norm(update) < avg_tol:
            break

    print("Average Newton iterations:", i)
    return x
