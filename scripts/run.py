import clarabel
import cvxpy
import numpy as np
import cvxpy as cp
import scipy
import scipy.sparse as sp
import madrona_stick as s

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

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

def clarabel_solve(A, v0, mu, penetrations, num_contact_pts):
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
    f = solver_results.x

    return np.array(f)

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


def cone_solve(M, bias, v, J, mu, penetrations, h, result):
    """
    Solves the objective function: (note that Af + v0 is v_c)
        min (1/2) * f^T A f + f^T v0 + q(Af + v0)
        subject to f \in K
    q is an imposed penalty function to prevent penetration
    """
    num_contacts_pts = int(J.shape[0] / 3)
    C = -bias
    M_inv = np.linalg.inv(M)

    # Sparse matrices
    M_sc = scipy.sparse.csc_matrix(M)
    J_sc = scipy.sparse.csc_matrix(J)
    A = J_sc @ M_inv @ J_sc.T
    v0 = J_sc @ (v + h * M_inv @ C)

    def q(v_c):
        # penalize where normal component of v_C is negative
        ind_neg = np.where(v_c < 0)
        ind_normal = np.arange(1, len(v_c), 3)
        ind_neg_normal = np.intersect1d(ind_neg, ind_normal)
        # squared loss
        return np.sum(v_c[ind_neg_normal] ** 2)

    def dq(v_c):
        out = np.zeros_like(v_c)
        # penalize where normal component of v_C is negative
        ind_neg = np.where(v_c < 0)
        ind_normal = np.arange(1, len(v_c), 3)
        ind_neg_normal = np.intersect1d(ind_neg, ind_normal)
        # gradient of squared loss
        out[ind_neg_normal] = 2 * v_c[ind_neg_normal]
        return out

    def hq(v_c):
        out = np.zeros((len(v_c), len(v_c)))
        # penalize where normal component of v_C is negative
        ind_neg = np.where(v_c < 0)
        ind_normal = np.arange(1, len(v_c), 3)
        ind_neg_normal = np.intersect1d(ind_neg, ind_normal)
        # hessian of squared loss
        out[ind_neg_normal, ind_neg_normal] = 2
        return out

    def obj(f):
        return 0.5 * f.T @ A @ f + f.T @ v0 + q(A @ f + v0)

    def d_obj(f):
        return A @ f + v0 + dq(A @ f + v0)

    def h_obj(f):
        return A + hq(A @ f + v0)

    if num_contacts_pts == 0:
        gen_forces = C
    else:
        f = scipy.optimize.minimize(obj, x0=np.zeros(num_contacts_pts * 3), jac=d_obj, hess=h_obj, method='trust-ncg')
        contact_imp = (J_sc.T @ f.x) / h
        gen_forces = C + contact_imp

    res = scipy.sparse.linalg.spsolve(M_sc, gen_forces)
    result[:] = res
    return

def mass_solve(M, bias, v, J, mu, penetrations, h, result):
    num_contacts_pts = int(J.shape[0] / 3)
    C = -bias
    M_inv = np.linalg.inv(M)

    # Sparse matrices
    M_sc = scipy.sparse.csc_matrix(M)
    J_sc = scipy.sparse.csc_matrix(J)
    A = J_sc @ M_inv @ J_sc.T
    v0 = J_sc @ (v + h * M_inv @ C)

    if num_contacts_pts == 0:
        gen_forces = C
    else:
        f = clarabel_solve(A, v0, mu, penetrations, num_contacts_pts)
        contact_imp = (J_sc.T @ f) / h
        gen_forces = C + contact_imp
    res = scipy.sparse.linalg.spsolve(M_sc, gen_forces)
    result[:] = res


if __name__ == "__main__":
    num_worlds = 1
    app = s.PhysicsApp(num_worlds)
    app.run(mass_solve)
