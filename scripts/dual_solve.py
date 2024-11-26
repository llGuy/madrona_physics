"""
Solves for the force f (dual problem) using hard cone constraints,
    but soft constraints elsewhere
"""
import numpy as np
import scipy.sparse as sp

from friction_cone import FrictionCones
from matrix_wrappers import AMatrix, HMatrix, MMatrix


def dual_solve(M, bias, v, J, mu, penetrations, h, result):
    """
    Solves the objective function: (note that Af + v0 is v_c)
        min (1/2) * f^T A f + f^T v0 + q(Af + v0)
        subject to f in K
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
        return A @ f + v0 + A @ dq(A @ f + v0)

    def h_obj(f):
        tmp = A @ hq(A @ f + v0)
        return HMatrix(A=A, E=A @ tmp.T).materialize()

    if num_contacts_pts == 0:
        gen_forces = C
    else:
        f0 = np.zeros(num_contacts_pts * 3)
        for i in range(num_contacts_pts):
            f0[i * 3] = 1.0
        # f = newton(obj, d_obj, h_obj, f0, 1e-5, 1e-8, cones)
        # f, info = trust_region_newton_cg(fun=obj, x0=f0, jac=d_obj, hess=h_obj, g_tol=1e-5)

        from scipy.optimize import minimize
        def ineq_constraint(f, mu):
            return np.array([f[3 * i] * mu[i] - np.linalg.norm(f[3 * i + 1: 3 * i + 3]) for i in range(len(mu))])

        res = minimize(obj, f0, jac=d_obj, hess=h_obj, method='SLSQP',
                       constraints={'type': 'ineq', 'fun': lambda f: ineq_constraint(f, mu)})

        f = res.x
        contact_imp = (J_sc.T @ f) / h
        gen_forces = C + contact_imp

    res = M_m.solve(gen_forces)
    result[:] = res
    return
