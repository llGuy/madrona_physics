import scipy.sparse as sp


class AMatrix:
    # A = J @ M^{-1} @ J.T, but rather than storing A, we store M and J
    def __init__(self, M, J):
        self.M = sp.csc_matrix(M)
        self.J = sp.csc_matrix(J)

    def matvec(self, x):
        return self.J @ sp.linalg.spsolve(self.M, self.J.T @ x)

    def dot(self, x):
        return self.matvec(x)

    def __matmul__(self, x):
        return self.matvec(x)


class HMatrix:
    def __init__(self, A, Hq):
        self.A = A
        self.Hq = Hq

    def matvec(self, x):
        return (self.A @ x) + self.Hq @ x

    def dot(self, x):
        return self.matvec(x)

    def __matmul__(self, x):
        return self.matvec(x)
