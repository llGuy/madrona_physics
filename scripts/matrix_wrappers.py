from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class MMatrix:
    def __init__(self, M):
        self.M = csc_matrix(M)

    def matvec(self, x):
        return self.M @ x

    def dot(self, x):
        return self.matvec(x)

    def __matmul__(self, x):
        return self.matvec(x)

    def solve(self, x):
        return spsolve(self.M, x)


class AMatrix:
    # A = J @ M^{-1} @ J.T, but rather than storing A, we store M and J
    def __init__(self, M, J):
        self.M = M
        self.J = csc_matrix(J)

    def matvec(self, x):
        return self.J @ self.M.solve(self.J.T @ x)

    def dot(self, x):
        return self.matvec(x)

    def __matmul__(self, x):
        return self.matvec(x)


class HMatrix:
    def __init__(self, A, Hq):
        self.A = A
        self.Hq = csc_matrix(Hq)

    def matvec(self, x):
        return (self.A @ x) + (self.Hq @ x)

    def dot(self, x):
        return self.matvec(x)

    def __matmul__(self, x):
        return self.matvec(x)
