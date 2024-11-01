import cvxpy
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import madrona_stick as s


def main_obj(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    return 0.5 * f.T @ (A @ f) + f.T @ v0


def constr1(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    s = 0
    for i in range(num_contact_pts):
        s += -kappa * np.log(f[3 * i])
    return s


def constr2(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    s = 0
    for i in range(num_contact_pts):
        s += np.log(mu[i] ** 2 * f[3 * i] ** 2 - f[3 * i + 1] ** 2 - f[3 * i + 2] ** 2)
    return -kappa * s


def constr3(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    s = 0
    vc = A @ f + v0
    for i in range(num_contact_pts):
        s += np.log(vc[3 * i] - penetrations[i])
    return -kappa * s


def d_main_obj(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    return A @ f + v0


def d_constr1(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    d = np.zeros(f.shape[0])
    for i in range(num_contact_pts):
        d[3 * i] = -kappa / f[3 * i]
    return d


def d_constr2(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    d = np.zeros(f.shape[0])
    for i in range(num_contact_pts):
        s = mu[i] ** 2 * f[3 * i] ** 2 - f[3 * i + 1] ** 2 - f[3 * i + 2] ** 2
        d[3 * i] = -2 * kappa * mu[i] ** 2 * f[3 * i] / s
        d[3 * i + 1] = 2 * kappa * f[3 * i + 1] / s
        d[3 * i + 2] = 2 * kappa * f[3 * i + 2] / s
    return d


def d_constr3(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    d = np.zeros(f.shape[0])
    vc = A @ f + v0
    for i in range(num_contact_pts):
        s = vc[3 * i] - penetrations[i]
        d += -kappa * A[3 * i, :] / s
    return d


def H_main_obj(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    return A


def H_constr1(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    H = np.zeros((f.shape[0], f.shape[0]))
    for i in range(num_contact_pts):
        H[3 * i, 3 * i] = kappa / (f[3 * i] ** 2)
    return H


def H_constr2(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    H = np.zeros((f.shape[0], f.shape[0]))
    for i in range(num_contact_pts):
        s = mu[i] ** 2 * f[3 * i] ** 2 - f[3 * i + 1] ** 2 - f[3 * i + 2] ** 2
        H[3 * i, 3 * i] = -2 * kappa * mu[i] ** 2 / s
        H[3 * i + 1, 3 * i + 1] = 2 * kappa / s
        H[3 * i + 2, 3 * i + 2] = 2 * kappa / s
    return H


def H_constr3(f, A, v0, mu, penetrations, num_contact_pts, kappa):
    H = np.zeros((f.shape[0], f.shape[0]))
    vc = A @ f + v0
    for i in range(num_contact_pts):
        s = vc[3 * i] - penetrations[i]
        H += kappa * np.outer(A[3 * i, :], A[3 * i, :]) / (s ** 2)
    return H


def scipy_solve(A, v0, mu, penetrations, result):
    num_contact_pts = result.shape[0] // 3
    if num_contact_pts == 0:
        return

    kappa = 1e-9

    def total_obj(f):
        return main_obj(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            constr1(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            constr2(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            constr3(f, A, v0, mu, penetrations, num_contact_pts, kappa)

    def total_d_obj(f):
        return d_main_obj(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            d_constr1(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            d_constr2(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            d_constr3(f, A, v0, mu, penetrations, num_contact_pts, kappa)

    def total_H_obj(f):
        return H_main_obj(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            H_constr1(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            H_constr2(f, A, v0, mu, penetrations, num_contact_pts, kappa) + \
            H_constr3(f, A, v0, mu, penetrations, num_contact_pts, kappa)

    clarabel_result = cvx_solve(A, v0, mu, penetrations, np.zeros(result.shape[0]))
    f0 = clarabel_result
    res = minimize(total_obj, f0, jac=total_d_obj, hess=total_H_obj, method='trust-ncg')
    result[:] = res.x


def cvx_solve(A, v0, mu, penetrations, result):
    num_contact_pts = result.shape[0] / 3
    if num_contact_pts == 0:
        return

    A_cpy = cvxpy.psd_wrap(A)
    R = 1e-8 * np.eye(result.shape[0])

    f = cp.Variable(result.shape[0])

    selection_mat = np.zeros((int(num_contact_pts), result.shape[0]))
    for row in range(int(num_contact_pts)):
        selection_mat[row][row * 3] = 1.0

    # Objective function: 0.5 * f.T @ (A + R) @ f + f.T @ v0
    objective = 0.5 * cp.quad_form(f, A_cpy + R) + f.T @ v0

    # Constraints
    constraints = [
        # Positivity constraints on f
        selection_mat @ f >= 0.0,

        # Positivity constraints on A @ f + v0
        selection_mat @ (A_cpy @ f + v0) >= penetrations
    ]
    #
    for contact in range(int(num_contact_pts)):
        constraints.append(cp.SOC(
            cp.multiply(mu[contact], f[contact * 3]),
            f[contact * 3 + 1: contact * 3 + 3])
        )

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    result[:] = f.value

    return f.value


if __name__ == "__main__":
    num_worlds = 1
    app = s.PhysicsApp(num_worlds)
    app.run(cvx_solve)
