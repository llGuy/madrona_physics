"""
Solves the reduced primal problem of finding the acceleration with soft constraints
"""
import numpy as np
import scipy.sparse as sp

from matrix_wrappers import MMatrix
from scripts.conjugate_gradient import nonlinear_cg
from scripts.newton import newton

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


def get_aref(v, J, r, h):
    """
    Computes the ``reference acceleration'' described by MuJoCo.
    The reference acceleration is based on Baumgarte stabilization
        (critically damped spring force for constraints)
    """
    time_const, damp_ratio = 2 * h, 1
    d_min, d_max, width, mid, power = 0.9, 0.95, 0.001, 0.5, 2

    imp_x = np.abs(r) / width
    imp_a = (1.0 / np.power(mid, power - 1)) * np.power(imp_x, power)
    imp_b = 1 - (1.0 / np.power(1 - mid, power - 1)) * np.power(1 - imp_x,
                                                                power)
    imp_y = np.where(imp_x < mid, imp_a, imp_b)
    imp = d_min + imp_y * (d_max - d_min)
    imp = np.clip(imp, d_min, d_max)
    imp = np.where(imp_x > 1.0, d_max, imp)

    k = 1 / (d_max * d_max * time_const * time_const * damp_ratio * damp_ratio)
    b = 2 / (d_max * time_const)

    aref = -b * (J @ v) - k * imp * r
    return aref


def reduced_primal(M, a_free, v, J, mus, penetrations, h, result):
    """
    Solves the reduced primal problem:
        min \|x - M^{-1} C\|_M^2 + s(Jx - a_ref)
    Where s is a convex function that encourages the input
        to be in the constrained space. For contacts,
        this is the dual of the friction cone
    """
    # Convert J from column-major to row-major
    J = J.T
    num_contacts_pts = int(J.shape[0] / 3)

    # Original matrices
    M_og = M
    J_og = J

    # Matrix wrappers
    M = MMatrix(M=M)
    J = sp.csc_matrix(J)

    # Compute reference acceleration
    r = np.zeros(J.shape[0])
    for i in range(num_contacts_pts):
        r[i * 3] = -penetrations[i]
    a_ref = get_aref(v, J, r, h)

    # Define convex s and our objective
    def s(jar):
        """
        Each dual cone is defined as:
            K^* = {p | 1 / mu * p_N >= |p_T|},
            the set of all valid accelerations

        Each contact is categorized into three zones:
            top zone:       p_N >= mu * |p_T|
            middle zone:    p_N < mu * |p_T| and mu * p_N + |p_T| > 0
            bottom zone:    mu * p_N + |p_T| <= 0

        For each contact point, the cost is:
            top zone:       0
            middle zone:    0.5 * (N - mu * T)^2
            bottom zone:    0.5 * (N^2 + T1^2 + T2^2)
        """
        cost = 0

        for i in range(len(jar) // 3):
            N, T1, T2, T, mu, mid_weight = get_norm_tangent_weights(jar, mus, i)
            # Top zone
            if N >= mu * T:
                pass
            # Bottom zone
            elif mu * N + T <= 0:
                cost += 0.5 * (N ** 2 + T1 ** 2 + T2 ** 2)
            # Middle zone
            else:
                cost += 0.5 * mid_weight * (N - mu * T) ** 2
        return cost

    def get_norm_tangent_weights(jar, ms, idx):
        N, T1, T2 = jar[3 * idx], jar[3 * idx + 1], jar[3 * idx + 2]
        T, mu = np.sqrt(T1 ** 2 + T2 ** 2), ms[idx]
        mid_weight = 1 / (1 + mu ** 2)
        return N, T1, T2, T, mu, mid_weight

    def ds(jar):
        out = np.zeros_like(jar)

        for i in range(len(jar) // 3):
            N, T1, T2, T, mu, mid_weight = get_norm_tangent_weights(jar, mus, i)
            if N >= mu * T:
                pass
            elif mu * N + T <= 0:
                out[3 * i] = N
                out[3 * i + 1] = T1
                out[3 * i + 2] = T2
            else:
                tmp = mid_weight * (N - mu * T)
                out[3 * i] = tmp
                out[3 * i + 1] = -tmp * mu * T1 / T
                out[3 * i + 2] = -tmp * mu * T2 / T
        return out

    def add_hessian_entry(
            rows: np.ndarray,
            cols: np.ndarray,
            data: np.ndarray,
            i: int, r: int, c: int, v: float):
        """
        Adds entry to the temporary storage for building the Hessian
        """
        rows[i], cols[i], data[i] = r, c, v

    def hs(jar):
        n_bottom, n_middle = 0, 0
        for i in range(len(jar) // 3):
            N, T1, T2, T, mu, mid_weight = get_norm_tangent_weights(jar, mus, i)
            if N >= mu * T:
                pass
            elif mu * N + T <= 0:
                n_bottom += 1
            else:
                n_middle += 1

        # Build sparse matrix the inexpensive way
        n_values = 3 * n_bottom + 9 * n_middle
        # These store the (row, column) indices and values of the Hessian
        rows, cols = np.zeros(n_values, dtype=np.uint32), np.zeros(n_values,
                                                                   dtype=np.uint32)
        data = np.zeros(n_values, dtype=np.float64)
        idx = 0  # How many temporary values we have stored

        for i in range(len(jar) // 3):
            N, T1, T2, T, mu, mid_weight = get_norm_tangent_weights(jar, mus, i)
            # Top zone
            if N >= mu * T:
                pass
            elif mu * N + T <= 0:
                N_idx, T1_idx, T2_idx = 3 * i, 3 * i + 1, 3 * i + 2
                add_hessian_entry(rows, cols, data, idx, N_idx, N_idx, 1)
                add_hessian_entry(rows, cols, data, idx + 1, T1_idx, T1_idx, 1)
                add_hessian_entry(rows, cols, data, idx + 2, T2_idx, T2_idx, 1)
                idx += 3
            else:
                W = mid_weight
                N_idx, T1_idx, T2_idx = 3 * i, 3 * i + 1, 3 * i + 2
                # Respect to N
                add_hessian_entry(rows, cols, data, idx, N_idx, N_idx, W)
                add_hessian_entry(rows, cols, data, idx + 1, N_idx, T1_idx,
                                  -W * mu * T1 / T)
                add_hessian_entry(rows, cols, data, idx + 2, N_idx, T2_idx,
                                  -W * mu * T2 / T)
                # Respect to T1
                add_hessian_entry(rows, cols, data, idx + 3, T1_idx, N_idx,
                                  -W * mu * T1 / T)
                add_hessian_entry(rows, cols, data, idx + 4, T1_idx, T1_idx,
                                  W * mu * (mu - ((N * T2 ** 2) / T ** 3)))
                add_hessian_entry(rows, cols, data, idx + 5, T1_idx, T2_idx,
                                  W * mu * (N * T1 * T2) / T ** 3)
                # Respect to T2
                add_hessian_entry(rows, cols, data, idx + 6, T2_idx, N_idx,
                                  -W * mu * T2 / T)
                add_hessian_entry(rows, cols, data, idx + 7, T2_idx, T1_idx,
                                  W * mu * (N * T1 * T2) / T ** 3)
                add_hessian_entry(rows, cols, data, idx + 8, T2_idx, T2_idx,
                                  W * mu * (mu - ((N * T1 ** 2) / T ** 3)))
                idx += 9

        hess = sp.csc_matrix((data, (rows, cols)),
                             shape=(jar.shape[0], jar.shape[0]))
        return hess

    def obj(x):
        x_min_a_free = x - a_free
        return 0.5 * x_min_a_free.T @ (M @ x_min_a_free) + s(J @ x - a_ref)

    def d_obj(x):
        x_min_a_free = x - a_free
        return (M @ x_min_a_free) + J.T @ ds(J @ x - a_ref)

    def h_obj(x):
        # return HMatrix(A=M, E=J.T @ hs(J @ x - a_ref) @ J)
        return M.M + J.T @ hs(J @ x - a_ref) @ J

    # Solve for x (\dot v)
    if num_contacts_pts == 0:
        result[:] = a_free
    else:
        # Tolerance of 1e-5 is around the lowest we can get with 32bit floats
        #   and given the condition number of M
        # a_solve = newton(df=d_obj, hess=h_obj, x0=a_free, tol=1e-5, M=M,
        #                  a_free=a_free, J=J, a_ref=a_ref, mus=mus)

        # Slower convergence but easier to implement
        tol, ls_tol = 1e-8, 0.01
        a_solve = nonlinear_cg(f=obj, df=d_obj, x0=a_free, tol=tol,
                               ls_tol=ls_tol, M=M, a_free=a_free, J=J,
                               a_ref=a_ref, mus=mus)
        result[:] = a_solve

    return
