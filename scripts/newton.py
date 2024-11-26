"""
Newton related solvers
"""
import numpy as np
from scipy.optimize import line_search


def inner_newton_cg(z0, g, r, d, H, tol, cg_max_iter):
    """
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



def newton(fun, df, hess, x0, tol, cg_tol):
    """
    Minimizes [fun] using Newton's method.
    The search direction [hess]^{-1}[df] is computed using the conjugate gradient method.
    """
    x = x0.copy()
    max_iter = len(x0) * 100
    cg_max_iter = len(x0) * 50
    f_old = fun(x)
    f_old_old = None
    for i in range(max_iter):
        # Check convergence first
        g = df(x)
        if np.linalg.norm(g) < tol:
            break

        # Newton step
        H = hess(x)
        z0 = np.zeros_like(x)
        r, d, p = g.copy(), -g, -g
        p, info = inner_newton_cg(z0=z0, g=g, r=r, d=d, H=H, tol=cg_tol, cg_max_iter=cg_max_iter)

        # Line search
        alpha, _, _, f_old, f_old_old, _ = line_search(fun, df, x, p, g, f_old, f_old_old)
        if alpha is None:
            print("Line search failed")  # why :(
            return x

        update = alpha * p
        x += update

        if np.linalg.norm(update) < tol:
            break

    return x
