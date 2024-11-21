import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import line_search

from scripts.friction_cone import FrictionCones
from scripts.matrix_wrappers import AMatrix, HMatrix, MMatrix


def inner_newton_cg(z0, g, r, d, H, tol, cg_max_iter):
    """
    Inner loop for Newton-CG, solves for the search direction p in the linear system
        Ap = -g, where A is the Hessian and g is the gradient.
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

def newton(fun, df, hess, x0, tol, cones):
    x = x0.copy()
    max_iter = len(x0) * 100
    cg_max_iter = len(x0) * 20
    f_old = fun(x)
    f_old_old = None
    for i in range(max_iter):
        # Newton step
        g, H = df(x), hess(x)

        z0 = np.zeros_like(x)
        r, d, p = g.copy(), -g, -g
        p, info = inner_newton_cg(z0=z0, g=g, r=r, d=d, H=H, tol=tol, cg_max_iter=cg_max_iter)

        # Line search
        alpha, _, _, f_old, f_old_old, _ = line_search(fun, df, x, p, g, f_old, f_old_old)
        if alpha is None:
            print("Line search failed")
            return x

        # Take the step if it keeps us in the cone
        proposed_update = alpha * p
        proposed_x = x + proposed_update
        if cones.in_cone(proposed_x):
            update = proposed_update
            x = proposed_x
        else:  # we must have left the cone at some point
            min_t = cones.get_min_t(x, p)
            update = min_t * p
            x += update

        if np.linalg.norm(update) < tol:
            break

    return x

def cone_solve(M, bias, v, J, mu, penetrations, h, result):
    """
    Solves the objective function: (note that Af + v0 is v_c)
        min (1/2) * f^T A f + f^T v0 + q(Af + v0)
        subject to f \in K
    q is an imposed penalty function to prevent penetration
    """
    num_contacts_pts = int(J.shape[0] / 3)
    # Matrix wrappers
    M_m = MMatrix(M=M)
    A = AMatrix(M=M_m, J=J)

    # Friction cones
    cones = FrictionCones(mus=mu)

    C = -bias
    J_sc = sp.csc_matrix(J)
    v0 = J_sc @ (v + h * M_m.solve(C))

    def q(v_c):
        # penalize where normal component of v_C is negative
        ind_neg = np.where(v_c < 0)
        ind_normal = np.arange(1, len(v_c), 3)
        ind_neg_normal = np.intersect1d(ind_neg, ind_normal)
        # squared loss
        return np.sum(v_c[ind_neg_normal] ** 2)

    def dq(v_c):
        out = np.zeros_like(v_c)
        # penalize where normal component of v_C is negative
        ind_neg = np.where(v_c < 0)
        ind_normal = np.arange(1, len(v_c), 3)
        ind_neg_normal = np.intersect1d(ind_neg, ind_normal)
        # gradient of squared loss
        out[ind_neg_normal] = 2 * v_c[ind_neg_normal]
        return out

    def hq(v_c):
        out = np.zeros((len(v_c), len(v_c)))
        # penalize where normal component of v_C is negative
        ind_neg = np.where(v_c < 0)
        ind_normal = np.arange(1, len(v_c), 3)
        ind_neg_normal = np.intersect1d(ind_neg, ind_normal)
        # hessian of squared loss
        out[ind_neg_normal, ind_neg_normal] = 2
        return out

    def obj(f):
        return 0.5 * f.T @ (A @ f) + f.T @ v0 + q(A @ f + v0)

    def d_obj(f):
        return A @ f + v0 + dq(A @ f + v0)

    def h_obj(f):
        return HMatrix(A, hq(A @ f + v0))

    if num_contacts_pts == 0:
        gen_forces = C
    else:
        f0 = np.zeros(num_contacts_pts * 3)
        for i in range(num_contacts_pts):
            f0[i * 3] = 1.0
        f = newton(obj, d_obj, h_obj, f0, 1e-3, cones)
        contact_imp = (J_sc.T @ f) / h
        gen_forces = C + contact_imp

    res = M_m.solve(gen_forces)
    result[:] = res
    return
