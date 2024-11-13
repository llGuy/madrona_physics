import clarabel
import cvxpy
import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import madrona_stick as s

def create_clarabel_matrices(n_contact_pts, mu, P, c, penetration):
    n_contact_pts = int(n_contact_pts)
    n_constraints = n_contact_pts + 3 * n_contact_pts + n_contact_pts

    n = 3 * n_contact_pts
    A = np.zeros((n_constraints, n))
    b = np.zeros(n_constraints)

    # x[3*i] ≥ 0
    for i in range(n_contact_pts):
        row = i
        col = 3*i
        A[row, col] = 1

    # Rearranging: (Px + c)[3*i] - penetration[i] ≥ 0
    # So: Px ≥ penetration - c
    start_idx = n_contact_pts
    for i in range(n_contact_pts):
        row = start_idx + i
        # Add the row of P corresponding to component 3*i
        A[row, :] = P[3*i, :]
        # Add the penetration constraint to b
        b[row] = c[3*i] - penetration[i]

    # [μ*x[3*i], x[3*i], x[3*i+1], x[3*i+2]]
    start_idx = n_contact_pts + n_contact_pts
    for i in range(n_contact_pts):
        A[start_idx + 3*i, 3*i] = mu[i]
        A[start_idx + 3*i + 1, 3*i + 1] = 1
        A[start_idx + 3*i + 2, 3*i + 2] = 1

    A *= -1
    # Create cone dictionary
    cones = []
    cones.append(clarabel.NonnegativeConeT(2*n_contact_pts))
    for i in range(n_contact_pts):
        cones.append(clarabel.SecondOrderConeT(3))

    return A, b, cones

def clarabel_solve(A, v0, mu, penetrations, result):
    num_contact_pts = result.shape[0] / 3
    if num_contact_pts == 0:
        return

    # Clarabel solves:
    # min x^T P x + c^T x ( for us, P = A, c = v0 )
    # s.t. Ax + s = b and s \in K
    P = A
    c = np.array(v0)
    A_c, b, cones = create_clarabel_matrices(num_contact_pts, mu, P, c, penetrations)

    # To csc matrix
    P = sp.csc_matrix(P)
    A_c = sp.csc_matrix(A_c)

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(P, c, A_c, b, cones, settings)
    solver_results = solver.solve()

    solver_results, status = solver_results, solver_results.status
    result[:] = solver_results.x

    return result

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
    # No penetration constraints
    for contact in range(int(num_contact_pts)):
        constraints.append(cp.SOC(
            cp.multiply(mu[contact], f[contact * 3]),
            f[contact * 3 + 1: contact * 3 + 3])
        )

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    result[:] = f.value

    return f.value

def mass_solve(M, tau, result):
    result[:] = np.linalg.solve(M, tau)


if __name__ == "__main__":
    num_worlds = 1
    app = s.PhysicsApp(num_worlds)
    app.run(mass_solve)
