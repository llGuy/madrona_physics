import numpy as np
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

    def materialize_inverse(self):
        return np.linalg.inv(self.M.toarray())

    def materialize(self):
        return self.materialize_inverse()


class AMatrix:
    # A = J @ M^{-1} @ J.T, but rather than storing A, we store M and J
    def __init__(self, M: MMatrix, J):
        self.M = M
        self.J = csc_matrix(J)

    def matvec(self, x):
        return self.J @ self.M.solve(self.J.T @ x)

    def dot(self, x):
        return self.matvec(x)

    def materialize(self):
        return self.J @ self.M.materialize_inverse() @ self.J.T

    def __matmul__(self, x):
        return self.matvec(x)


class HMatrix:
    """
    Wrapper for Hessian matrix of the form A + E, where A is a wrapper
    """
    def __init__(self, A, E):
        self.A = A
        self.E = csc_matrix(E)

    def matvec(self, x):
        return (self.A @ x) + (self.E @ x)

    def dot(self, x):
        return self.matvec(x)

    def __matmul__(self, x):
        return self.matvec(x)

    def materialize(self):
        return self.A.materialize() + self.E.toarray()