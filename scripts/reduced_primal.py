import numpy as np
import scipy.sparse as sp

from matrix_wrappers import MMatrix


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
        Helper function for computing the mu * normal and \|tangent\|
        """
        in_normal = np.zeros_like(jar, dtype=bool)
        in_normal[0::3] = True

        jar_normal = jar[in_normal]
        jar_tangent = jar[~in_normal].reshape(-1, 2)
        jar_tangent = np.linalg.norm(jar_tangent, axis=1)
        return jar_normal, jar_tangent

    def compute_zones(normal, tangent, assert_check=False):
        mu_normal = np.multiply(mu, normal)
        mu_tangent = np.multiply(mu, tangent)
        inv_mu_normal = np.multiply(1 / mu, normal)

        # Top zone (inside dual cone)
        ind_top = np.where(inv_mu_normal >= tangent)[0]

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
        weight = 10 * np.ones(jar.shape[0])  # TODO define these

        # Get the weighting for the middle zone
        in_normal = np.zeros_like(jar, dtype=bool)
        in_normal[0::3] = True
        weight_normal = weight[in_normal]

        # TODO: check, we want continuity of s for bottom -> middle
        middle_weight = weight_normal / (mu ** 2 * (1 + mu ** 2))
        # middle_weight = weight_normal / (1 + mu ** 2)
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
        # Cost for bottom zone
        for ib in ind_bottom:
            N, T1, T2 = jar[3 * ib], jar[3 * ib + 1], jar[3 * ib + 2]
            cost += 0.5 * weight[ib] * (N ** 2 + T1 ** 2 + T2 ** 2)

        # Cost for middle zone is quadratic in (N - mu * T)
        for im in ind_middle:
            N, T1, T2 = jar[3 * im], jar[3 * im + 1], jar[3 * im + 2]
            T = np.linalg.norm([T1, T2])
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
            out[3 * ib] += weight[ib] * N
            out[3 * ib + 1] += weight[ib] * T1
            out[3 * ib + 2] += weight[ib] * T2

        # Middle zone: Cost is (1/2) * D_middle * (N - mu * T)^2
        #  d/dN (cost) = D_middle * (N - mu * T)
        #  d/dT_i (cost = -D_middle * mu*T_i * (N - mu * T) / T
        for im in ind_middle:
            N, T1, T2 = jar[3 * im], jar[3 * im + 1], jar[3 * im + 2]
            T = np.linalg.norm([T1, T2])
            out[3 * im] += middle_weight[im] * (N - mu[im] * T)
            out[3 * im + 1] += -middle_weight[im] * mu[im] * T1 * (N - mu[im] * T) / T
            out[3 * im + 2] += -middle_weight[im] * mu[im] * T2 * (N - mu[im] * T) / T

        return out

    def hs(jar):
        raise NotImplementedError

    def obj(x):
        x_min_a_free = x - a_free
        return a_free.T @ (M @ x_min_a_free) + s(J @ x - a_ref)

    def d_obj(x):
        x_min_a_free = x - a_free
        return 2 * (M @ x_min_a_free) + J.T @ ds(J @ x - a_ref)

    def h_obj(x):
        # TODO: eventually implement hs and make new wrapper
        # 2 * M + J.T @ hs(J @ x - a_ref) @ J
        raise NotImplementedError

    # Solve for x (\dot v)
    if num_contacts_pts == 0:
        result[:] = a_free
    else:
        # TODO: run optimization
        import scipy
        a_solve = scipy.optimize.minimize(obj, a_free, jac=d_obj, method="BFGS")
        # result[:] = scipy.linalg.pinv(J_og) @ a_ref
        result[:] = a_solve.x

    return
