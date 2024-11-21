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
        subject to f in K
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

    def s(jar):
        raise NotImplementedError

    def ds(jar):
        raise NotImplementedError

    def hq(jar):
        raise NotImplementedError

    def obj(x):
        x_a_free = x - a_free
        return a_free.T @ (M @ x_a_free) + s(J @ x - a_ref)

    def d_obj(x):
        x_a_free = x - a_free
        return 2 * (M @ x_a_free) + J.T @ ds(J @ x - a_ref)

    def h_obj(x):
        return 2 * M + J.T @ hq(J @ x - a_ref) @ J

    if num_contacts_pts == 0:
        result[:] = a_free
    else:
        import scipy
        result[:] = scipy.linalg.pinv(J_og) @ a_ref

    return
