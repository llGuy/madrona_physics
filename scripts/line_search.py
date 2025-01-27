"""
Exact line search method
"""
import numpy as np

def exact_line_search(xk, pk, tol, M, a_free, J, J_e, a_ref, a_e_ref, mus, precision):
    """
    Computes the alpha that minimizes phi(alpha) = f(xk + alpha * pk)
    """
    # Search vector too small
    if np.linalg.norm(pk) < 1e-15:
        return 0

    # Pre-compute some values
    x_min_a_free = xk - a_free
    # M @ (x - a_free)
    Mx_min_a_free = M @ x_min_a_free
    # p.T @ M @ p
    pMp = np.dot(pk, M @ pk)
    # p.T @ M @ (x - a_free)
    pMx_free = np.dot(pk, Mx_min_a_free)
    # (x - a_free).T @ M @ (x - a_free)
    x_min_M_x_min = np.dot(x_min_a_free, Mx_min_a_free)
    # J @ x - a_ref
    Jx_aref = J @ xk - a_ref
    # J @ p
    Jp = J @ pk

    Jex_aref = J_e @ xk - a_e_ref
    Jep = J_e @ pk

    def fdh_phi(a):
        """
        Computes the function evaluation and the first and second derivatives
        of the line search function
        """
        # Process Gauss first
        fun = 0.5 * a ** 2 * pMp + a * pMx_free + 0.5 * x_min_M_x_min
        grad = a * pMp + pMx_free
        hess = pMp

        # Then process cones
        for idx in range(len(Jx_aref) // 3):
            # Original
            N, T1, T2 = Jx_aref[3 * idx], Jx_aref[3 * idx + 1], Jx_aref[
                3 * idx + 2]
            mu = mus[idx]
            mw = 1 / (1 + mu ** 2)

            # After search
            p0, p1, p2 = Jp[3 * idx], Jp[3 * idx + 1], Jp[3 * idx + 2]
            Np = N + a * p0
            T1p = T1 + a * p1
            T2p = T2 + a * p2
            Tp = np.sqrt(T1p ** 2 + T2p ** 2)

            # Top zone
            if Np >= mu * Tp:
                pass

            # Bottom zone
            elif mu * Np + Tp <= 0:
                p_sq = p0 ** 2 + p1 ** 2 + p2 ** 2
                fun += Np ** 2 + Tp ** 2
                grad += p0 * N + p1 * T1 + p2 * T2 + a * p_sq
                hess += p_sq

            # Middle zone
            else:
                # dN' / d alpha
                dNp_da = p0
                # dTp' / d alpha
                dTp_da = (p1 * T1 + p2 * T2 + a * (p1 ** 2 + p2 ** 2)) / Tp
                # d^2 Tp' / d alpha^2
                d2Tp_da2 = (p2 * T1 - p1 * T2) ** 2 / Tp ** 3
                # Derivative of (Np - mu * Tp) wrst alpha
                tmp = Np - mu * Tp
                d_tmp = dNp_da - mu * dTp_da
                fun += mw * tmp ** 2
                grad += mw * tmp * d_tmp
                hess += mw * (d_tmp ** 2 + tmp * (-mu * d2Tp_da2))

        for idx in range(len(Jex_aref)):
            # Original
            orig = Jex_aref[idx]
            # After search
            dj = Jep[idx]

            Np = orig + a * dj

            fun += Np ** 2
            grad += orig * dj + a * (dj ** 2)
            hess += dj ** 2

        return fun, grad, hess

    alpha = np.array(0, dtype=precision)
    f_alpha, d_alpha, h_alpha = fdh_phi(alpha)
    alpha1 = alpha - d_alpha / h_alpha  # Newton step
    f_alpha1, _, _ = fdh_phi(alpha1)
    if f_alpha < f_alpha1:
        alpha1 = alpha

    _, d_alpha1, h_alpha1 = fdh_phi(alpha1)
    # initial convergence
    if np.abs(d_alpha1) < tol:
        return alpha1

    # Opposing direction of gradient at alpha1
    a_dir = 1 if d_alpha1 < 0 else -1
    ls_iters = 50
    for i in range(ls_iters):
        _, d_alpha1, h_alpha1 = fdh_phi(alpha1)
        # gradient moves in the opposite direction as alpha1, start bracketing
        if d_alpha1 * a_dir > -tol:
            break
        # Converged
        if np.abs(d_alpha1) < tol:
            return alpha1

        # Newton step
        alpha1 -= d_alpha1 / h_alpha1
    else:
        print("Failure to bracket")
        return alpha1

    # Bracketing to find where d_phi equals zero
    alpha_low = alpha1
    alpha_high = alpha1 - d_alpha1 / h_alpha1

    _, d_alpha_low, _ = fdh_phi(alpha_low)
    if d_alpha_low > 0:
        alpha_low, alpha_high = alpha_high, alpha_low

    for i in range(ls_iters):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        _, d_alpha_mid, _ = fdh_phi(alpha_mid)
        if np.abs(d_alpha_mid) < tol:
            return alpha_mid

        # Narrow the bracket
        if d_alpha_mid > 0:
            alpha_high = alpha_mid
        else:
            alpha_low = alpha_mid

        # Bracketing is small
        if np.abs(alpha_high - alpha_low) < tol:
            return alpha_mid
    else:
        print("Failure to converge")
        return alpha_mid
