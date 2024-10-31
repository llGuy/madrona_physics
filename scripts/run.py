import numpy as np
import cvxpy as cp
import madrona_stick as s

def create_symmetric_a_decomp(A):
    evalues, evectors = np.linalg.eig(A)
    lam_sqrt = np.diag(np.sqrt(evalues))
    # test = evectors @ lam @ evectors.T
    return evectors @ lam_sqrt

def cvx_solve(A, v0, mu, result):
    num_contact_pts = result.shape[0] / 3

    if num_contact_pts == 0:
        return

    A_part = create_symmetric_a_decomp(A)

    A_cpy = A_part @ A_part.T

    # A_cpy = cp.Parameter(A.shape, symmetric=True)
    # A_cpy.value = A_part @ A_part.T

    f = cp.Variable(result.shape[0])

    selection_mat = np.zeros((int(num_contact_pts), result.shape[0]));
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

    for contact in range(int(num_contact_pts)):
        constraints.append(cp.SOC(cp.multiply(mu[contact], f[contact * 3]),
                                  f[contact * 3 + 1 : (contact + 1) * 3]))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    print(f.value)

num_worlds = 1
app = s.PhysicsApp(num_worlds)
app.run(cvx_solve)
