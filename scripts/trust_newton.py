import numpy as np


def compute_tau(z, d, trust_radius):
    """
    Solve the scalar quadratic equation ||z + t d|| == trust_radius.
    This is like a line-sphere intersection.
    Return the two values of t, ta and tb, such that ta <= tb.
    """
    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius ** 2

    sqrt_discriminant = np.sqrt(b * b - 4 * a * c)
    aux = b + np.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return sorted([ta, tb])


def tr_predict(f0, p, g, A):
    """
    Predict the value of the objective function at the next step
    """
    return f0 + np.dot(g, p) + 0.5 * np.dot(p, A.dot(p))


def tr_solve(f0, x, g, A, max_iter, tol, trust_radius):
    """
    Use conjugate gradient to solve for the search direction p in the trust region.
    """
    # Check if we are already at a minimum
    z0 = np.zeros_like(x)
    if np.linalg.norm(g) <= tol:
        return z0, 0

    # Initialize
    z = z0
    r = g
    d = -r

    # Iterate to solve for the search direction
    for j in range(max_iter):
        Ad = A.dot(d)
        dAd = np.dot(d, Ad)
        # Direction of non-positive curvature
        if dAd <= 0:
            # Compute the two points that intersect the trust region boundary
            ta, tb = compute_tau(z, d, trust_radius)
            pa, pb = z + ta * d, z + tb * d
            # Choose the direction with the lowest predicted objective function value
            if tr_predict(f0, pa, g, A) < tr_predict(f0, pb, g, A):
                return pa, 1
            else:
                return pb, 1

        # Otherwise, continue with the conjugate gradient
        r_squared = np.dot(r, r)
        alpha = r_squared / dAd
        z_next = z + alpha * d
        # Move back to the boundary of the trust region
        if np.linalg.norm(z_next) >= trust_radius:
            # We require tau >= 0, take the positive root
            _, tau = compute_tau(z, d, trust_radius)
            p = z + tau * d
            return p, 1

        # Update residual, check for convergence
        r_next = r + alpha * Ad
        r_next_squared = np.dot(r_next, r_next)
        if np.sqrt(r_next_squared) < tol:
            return z_next, 0
        beta = r_next_squared / r_squared
        d_new = -r_next + beta * d

        z = z_next
        r = r_next
        d = d_new

    else:
        return z, 2


def trust_region_newton_cg(fun, x0, jac, hess, g_tol=1e-8):
    # --- Initialize trust-region ---
    trust_radius = 1.0
    max_trust_radius = 1000.0
    eta = 0.15

    # Max iterations for outer loop and then inner conjugate gradient loop
    max_iter = len(x0) * 100
    cg_max_iter = len(x0) * 200

    x = x0
    f_old = fun(x)

    for k in range(max_iter):
        g = jac(x)
        g_mag = np.linalg.norm(g)
        # Termination condition
        if g_mag < g_tol:
            return x, 0

        # Try an initial step and trusting it (conjugate gradient to solve the sub-problem)
        A = hess(x)
        cg_tol = min(0.5, np.sqrt(g_mag)) * g_mag
        p, hit_boundary = tr_solve(f0=f_old, x=x, g=g, A=A, max_iter=cg_max_iter, tol=cg_tol,
                                   trust_radius=trust_radius)
        # Did not find a valid step
        if hit_boundary == 2:
            return x, 2
        x_new = x + p
        f_new = fun(x_new)

        # Actual reduction and predicted reduction
        df = f_old - f_new
        # Predicted reduction by the quadratic model
        df_pred = f_old - tr_predict(f_old, p, g, A)

        # Updating trust region radius
        rho = df / df_pred
        if rho < 0.25:  # poor prediction, reduce trust radius
            trust_radius *= 0.25
        elif rho > 0.75 and hit_boundary:  # good step and on the boundary, can increase radius
            trust_radius = min(2 * trust_radius, max_trust_radius)

        # Accept step
        if rho > eta:
            x = x_new
            f_old = f_new

    return x, 1
