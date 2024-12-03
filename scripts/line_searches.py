import numpy as np

def my_line_search(f, df, xk, pk):
    c1, c2 = 1e-4, 0.9
    max_iter = 100
    alpha = 1
    alpha_prev = 0

    def phi(a):
        return f(xk + a * pk)

    def d_phi(a):
        return np.dot(df(xk + a * pk), pk)

    phi0 = phi(0)
    d_phi0 = d_phi(0)
    phi_prev = phi0

    for i in range(max_iter):
        phi_i = phi(alpha)
        if (phi_i > phi0 + c1 * alpha * d_phi0
                or (i > 0 and phi_i >= phi_prev)):
            return zoom(alpha_prev, alpha, f, df, xk, pk)

        d_phi_i = d_phi(alpha)
        if abs(d_phi_i) <= -c2 * d_phi0:
            return alpha

        if d_phi_i >= 0:
            return zoom(alpha, alpha_prev, f, df, xk, pk)

        # Update and choose next alpha between [alpha, alpha_max]
        alpha_prev = alpha
        phi_prev = phi_i
        alpha = 2 * alpha

    return

def zoom(alpha_lo, alpha_hi, f, df, xk, pk):
    c1, c2 = 1e-4, 0.9
    max_iter = 10

    def phi(a):
        return f(xk + a * pk)

    def d_phi(a):
        return np.dot(df(xk + a * pk), pk)

    phi_zero = phi(0)
    d_phi_zero = d_phi(0)

    for i in range(max_iter):
        alpha_j = (alpha_lo + alpha_hi) / 2
        phi_j = phi(alpha_j)
        if (phi_j > phi_zero + c1 * alpha_j * d_phi_zero
                or phi_j >= phi(alpha_lo)):
            alpha_hi = alpha_j
        else:
            d_phi_j = d_phi(alpha_j)
            if abs(d_phi_j) <= -c2 * d_phi_zero:
                return alpha_j
            if d_phi_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
    return
