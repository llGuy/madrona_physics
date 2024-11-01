import cvxpy
import numpy as np
import cvxpy as cp
import madrona_stick as s

def prepare_A(A):
    # Force symmetry
    A_cpy = np.triu(A)
    A_cpy = A_cpy + A_cpy.T - np.diag(np.diag(A_cpy))
    # Boost eigenvalues, tell cvxpy that A is PSD
    A_cpy += 1e-8 * np.eye(A_cpy.shape[0])
    A_cpy = cvxpy.psd_wrap(A_cpy)
    return A_cpy


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
        selection_mat @ f>= 0.0,

        # Positivity constraints on A @ f + v0
        selection_mat @ (A_cpy @ f + v0) >= 0.0
    ]
    #
    for contact in range(int(num_contact_pts)):
        constraints.append(cp.SOC(cp.multiply(mu[contact], f[contact * 3]),
                                  f[contact * 3 + 1 : (contact + 1) * 3]))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    print(f.value)
    result[:] = f.value

num_worlds = 1
app = s.PhysicsApp(num_worlds)
app.run(cvx_solve)
