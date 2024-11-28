"""
Solves the reduced primal problem of finding the acceleration with soft constraints
"""
import numpy as np
import scipy.sparse as sp

from matrix_wrappers import MMatrix
from matrix_wrappers import HMatrix
from newton import newton


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
    imp_b = 1 - (1.0 / np.power(1 - mid, power - 1)) * np.power(1 - imp_x, power)
    imp_y = np.where(imp_x < mid, imp_a, imp_b)
    imp = d_min + imp_y * (d_max - d_min)
    imp = np.clip(imp, d_min, d_max)
    imp = np.where(imp_x > 1.0, d_max, imp)

    k = 1 / (d_max * d_max * time_const * time_const * damp_ratio * damp_ratio)
    b = 2 / (d_max * time_const)

    aref = -b * (J @ v) - k * imp * r
    return aref


def reduced_primal(M, bias, v, J, mu, penetrations, h, result):
    """
    Solves the reduced primal problem:
        min \|x - M^{-1} C\|_M^2 + s(Jx - a_ref)
    Where s is a convex function that encourages the input
        to be in the constrained space. For contacts,
        this is the dual of the friction cone
    """
    num_contacts_pts = int(J.shape[0] / 3)

    # Original matrices
    M_og = M
    J_og = J

    # Matrix wrappers
    M = MMatrix(M=M)
    J = sp.csc_matrix(J)

    # Compute unconstrained acceleration
    C = -bias
    a_free = M.solve(C)

    # Compute reference acceleration
    r = np.zeros(J.shape[0])
    for i in range(num_contacts_pts):
        r[i * 3] = -penetrations[i]
    a_ref = get_aref(v, J, r, h)

    # Define convex s and our objective
    def get_normal_tangent(jar):
        """
        Helper function for computing normal and \|tangent\|
        """
        in_normal = np.zeros_like(jar, dtype=bool)
        in_normal[0::3] = True

        jar_normal = jar[in_normal]
        jar_tangent = jar[~in_normal].reshape(-1, 2)
        jar_tangent = np.linalg.norm(jar_tangent, axis=1)
        return jar_normal, jar_tangent

    def compute_zones(normal, tangent, assert_check=False):
        """
        Returns the indices for [normal, tangent] for the
            top, bottom, and middle zones defined as:
        top zone:       p_N >= mu * |p_T|
        middle zone:    p_N < mu * |p_T| and mu * p_N + |p_T| > 0
        bottom zone:    mu * p_N + |p_T| <= 0
        """
        mu_normal = np.multiply(mu, normal)
        mu_tangent = np.multiply(mu, tangent)

        # Top zone (inside dual cone)
        ind_top = np.where(normal >= mu_tangent)[0]

        # Bottom zone (reflection of K across x-axis)
        ind_bottom = np.where(mu_normal + tangent <= 0)[0]

        # Middle zone (everything else)
        ind_middle = np.where((normal < mu_tangent) & (mu_normal + tangent > 0))[0]

        # No overlap between the zones, covers all cases
        if assert_check:
            assert np.intersect1d(ind_top, ind_bottom).size == 0
            assert np.union1d(ind_top, np.union1d(ind_bottom, ind_middle)).size == normal.shape[0]

        return ind_top, ind_bottom, ind_middle

    def get_contact_weights(jar):
        """
        Computes weights for each contact component
        """
        # Weights for each contact component
        weight = 1 * np.ones(jar.shape[0])  # TODO define these

        # Get the weighting for the middle zone
        in_normal = np.zeros_like(jar, dtype=bool)
        in_normal[0::3] = True
        weight_normal = weight[in_normal]

        # TODO: check, we want continuity of s for bottom -> middle
        # middle_weight = weight_normal / (mu ** 2 * (1 + mu ** 2))
        middle_weight = weight_normal / (1 + mu ** 2)
        return weight, middle_weight

    def s(jar):
        """
        Each dual cone is defined as:
            K^* = {p | 1 / mu * p_N >= |p_T|}
        """
        normal, tangent = get_normal_tangent(jar)
        ind_top, ind_bottom, ind_middle = compute_zones(normal, tangent)
        weight, middle_weight = get_contact_weights(jar)

        cost = 0
        # Cost for bottom zone is sum of squares
        for ib in ind_bottom:
            N, T1, T2 = jar[3 * ib], jar[3 * ib + 1], jar[3 * ib + 2]
            WN, WT1, WT2 = weight[3 * ib], weight[3 * ib + 1], weight[3 * ib + 2]
            cost += 0.5 * (WN * N ** 2 + WT1 * T1 ** 2 + WT2 * T2 ** 2)

        # Cost for middle zone is quadratic in (N - mu * T)
        for im in ind_middle:
            N, T1, T2 = jar[3 * im], jar[3 * im + 1], jar[3 * im + 2]
            T = np.sqrt(T1 ** 2 + T2 ** 2)
            cost += 0.5 * middle_weight[im] * (N - mu[im] * T) ** 2
        return cost

    def ds(jar):
        normal, tangent = get_normal_tangent(jar)
        ind_top, ind_bottom, ind_middle = compute_zones(normal, tangent)
        weight, middle_weight = get_contact_weights(jar)

        out = np.zeros_like(jar)

        # Bottom zone: Cost is (1/2) * D * (N^2 + T_1^2 + T_2^2)
        #   d/dN (cost) = D * N
        #   d/dT_i (cost) = D * T_i
        for ib in ind_bottom:
            N, T1, T2 = jar[3 * ib], jar[3 * ib + 1], jar[3 * ib + 2]
            WN, WT1, WT2 = weight[3 * ib], weight[3 * ib + 1], weight[3 * ib + 2]
            out[3 * ib] = WN * N
            out[3 * ib + 1] = WT1 * T1
            out[3 * ib + 2] = WT2 * T2

        # Middle zone: Cost is (1/2) * D_middle * (N - mu * T)^2
        #  d/dN (cost) = D_middle * (N - mu * T)
        #  d/dT_i (cost = -D_middle * mu*T_i * (N - mu * T) / T
        for im in ind_middle:
            N, T1, T2 = jar[3 * im], jar[3 * im + 1], jar[3 * im + 2]
            T = np.sqrt(T1 ** 2 + T2 ** 2)
            W = middle_weight[im]
            tmp = W * (N - mu[im] * T)
            out[3 * im] = tmp
            out[3 * im + 1] = -tmp * mu[im] * T1 / T
            out[3 * im + 2] = -tmp * mu[im] * T2 / T

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
        normal, tangent = get_normal_tangent(jar)
        ind_top, ind_bottom, ind_middle = compute_zones(normal, tangent)
        weight, middle_weight = get_contact_weights(jar)

        # Build sparse matrix the inexpensive way
        n_values = 3 * ind_bottom.size + 9 * ind_middle.size
        # These store the (row, column) indices and values of the Hessian
        rows, cols = np.zeros(n_values, dtype=np.uint32), np.zeros(n_values, dtype=np.uint32)
        data = np.zeros(n_values, dtype=np.float64)
        idx = 0  # How many temporary values we have stored

        # Bottom zone: Cost is (1/2) * D * (N^2 + T_1^2 + T_2^2)
        #   Mixed derivatives are zero
        for ib in ind_bottom:
            N_idx, T1_idx, T2_idx = 3 * ib, 3 * ib + 1, 3 * ib + 2
            add_hessian_entry(rows, cols, data, idx, N_idx, N_idx, weight[N_idx])
            add_hessian_entry(rows, cols, data, idx + 1, T1_idx, T1_idx, weight[T1_idx])
            add_hessian_entry(rows, cols, data, idx + 2, T2_idx, T2_idx, weight[T2_idx])
            idx += 3

        # Middle zone: Cost is (1/2) * D_middle * (N - mu * T)^2
        for im in ind_middle:
            N_idx, T1_idx, T2_idx = 3 * im, 3 * im + 1, 3 * im + 2
            N, T1, T2 = jar[N_idx], jar[T1_idx], jar[T2_idx]
            T = np.sqrt(T1 ** 2 + T2 ** 2)
            W = middle_weight[im]
            # Respect to N
            add_hessian_entry(rows, cols, data, idx, N_idx, N_idx, W)
            add_hessian_entry(rows, cols, data, idx + 1, N_idx, T1_idx, -W * mu[im] * T1 / T)
            add_hessian_entry(rows, cols, data, idx + 2, N_idx, T2_idx, -W * mu[im] * T2 / T)
            # Respect to T1
            add_hessian_entry(rows, cols, data, idx + 3, T1_idx, N_idx, -W * mu[im] * T1 / T)
            add_hessian_entry(rows, cols, data, idx + 4, T1_idx, T1_idx,
                              W * mu[im] * (mu[im] - ((N * T2 ** 2) / T ** 3)))
            add_hessian_entry(rows, cols, data, idx + 5, T1_idx, T2_idx, W * mu[im] * (N * T1 * T2) / T ** 3)
            # Respect to T2
            add_hessian_entry(rows, cols, data, idx + 6, T2_idx, N_idx, -W * mu[im] * T2 / T)
            add_hessian_entry(rows, cols, data, idx + 7, T2_idx, T1_idx, W * mu[im] * (N * T1 * T2) / T ** 3)
            add_hessian_entry(rows, cols, data, idx + 8, T2_idx, T2_idx,
                              W * mu[im] * (mu[im] - ((N * T1 ** 2) / T ** 3)))
            idx += 9

        hess = sp.csc_matrix((data, (rows, cols)), shape=(jar.shape[0], jar.shape[0]))
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
        a_solve = newton(fun=obj, df=d_obj, hess=h_obj, x0=a_free, tol=1e-5)
        result[:] = a_solve

    # print(result)

    return
