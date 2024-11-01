import cvxpy
import numpy as np
import cvxpy as cp
import madrona_stick as s
from scipy.optimize import minimize


def prepare_A(A):
    # Force symmetry
    A_cpy = np.triu(A)
    A_cpy = A_cpy + A_cpy.T - np.diag(np.diag(A_cpy))
    # Boost eigenvalues, tell cvxpy that A is PSD
    A_cpy += 1e-8 * np.eye(A_cpy.shape[0])
    A_cpy = cvxpy.psd_wrap(A_cpy)
    return A_cpy


def main_obj(f, A, v0, mu, num_contact_pts, kappa):
    return 0.5 * f.T @ (A @ f) + f.T @ v0


def constr1(f, A, v0, mu, num_contact_pts, kappa):
    s = 0
    for i in range(num_contact_pts):
        s += np.log(f[3 * i])
    return -kappa * s


def constr2(f, A, v0, mu, num_contact_pts, kappa):
    s = 0
    for i in range(num_contact_pts):
        s += np.log(mu[3 * i] ** 2 * f[3 * i] ** 2 - f[3 * i + 1] ** 2 - f[3 * i + 2] ** 2)
    return -kappa * s


def constr3(f, A, v0, mu, num_contact_pts, kappa):
    s = 0
    for i in range(num_contact_pts):
        s += np.log((A @ f + v0)[3 * i])
    return -kappa * s


def d_main_obj(f, A, v0, mu, num_contact_pts, kappa):
    return A @ f + v0


def d_constr1(f, A, v0, mu, num_contact_pts, kappa):
    d = np.zeros(f.shape[0])
    for i in range(num_contact_pts):
        d[3 * i] = -kappa * 1 / f[3 * i]
    return d


def d_constr2(f, A, v0, mu, num_contact_pts, kappa):
    d = np.zeros(f.shape[0])
    for i in range(num_contact_pts):
        s = mu[3 * i] ** 2 * f[3 * i] ** 2 - f[3 * i + 1] ** 2 - f[3 * i + 2] ** 2
        d[3 * i] = -kappa * 2 * mu[3 * i] ** 2 * f[3 * i] / s
        d[3 * i + 1] = kappa * 2 * f[3 * i + 1] / s
        d[3 * i + 2] = kappa * 2 * f[3 * i + 2] / s
    return d


def d_constr3(f, A, v0, mu, num_contact_pts, kappa):
    d = np.zeros(f.shape[0])
    vc = A @ f + v0
    for i in range(num_contact_pts):
        s = vc[3 * i]
        d += -kappa * A[3 * i, :] / s
    return d


def scipy_solve(A, v0, mu, result):
    num_contact_pts = result.shape[0] / 3
    if num_contact_pts == 0:
        return

    kappa= 1e-6
    f0 = np.zeros(result.shape[0])
    for i in range(int(num_contact_pts)):
        f0[3 * i] = 1000
    res = minimize(main_obj, x0=f0,
                   args=(A, v0, mu, num_contact_pts, kappa),
                   jac=d_main_obj,
                   method='BFGS')
    print(res.x)
    result[:] = res.x

def cvx_solve(A, v0, mu, result):
    num_contact_pts = result.shape[0] / 3
    if num_contact_pts == 0:
        return

    A_cpy = prepare_A(A)

    f = cp.Variable(result.shape[0])

    selection_mat = np.zeros((int(num_contact_pts), result.shape[0]))
    for row in range(int(num_contact_pts)):
        selection_mat[row][row * 3] = 1.0

    # Objective function: 0.5 * f.T @ A @ f + f.T @ v0
    objective = 0.5 * cp.quad_form(f, A_cpy) + f.T @ v0

    # Constraints
    constraints = [
        # Positivity constraints on f
        selection_mat @ f >= 0.0,

        # Positivity constraints on A @ f + v0
        selection_mat @ (A_cpy @ f + v0) >= 0.0
    ]
    #
    for contact in range(int(num_contact_pts)):
        constraints.append(cp.SOC(
            cp.multiply(mu[contact], f[contact * 3]),
            f[contact * 3 + 1: contact * 3 + 3])
        )

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    print(f.value)
    result[:] = f.value


num_worlds = 1
app = s.PhysicsApp(num_worlds)
app.run(cvx_solve)
