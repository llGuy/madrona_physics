import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


class MatrixWrapper:
    def matvec(self, x):
        raise NotImplementedError

    def dot(self, x):
        raise NotImplementedError

    def __matmul__(self, x):
        return self.matvec(x)

    def materialize(self):
        raise NotImplementedError


class MMatrix(MatrixWrapper):
    def __init__(self, M):
        self.M = csc_matrix(M)
        self.LU = splu(self.M)  # Eventually should be a Cholesky factorization

    def matvec(self, x):
        return self.M @ x

    def dot(self, x):
        return self.matvec(x)

    def solve(self, x):
        # convert x to float32
        x = x.astype(np.float32)
        return self.LU.solve(x)

    def materialize_inverse(self):
        return np.linalg.inv(self.M.toarray())

    def materialize(self):
        return self.materialize_inverse()


class AMatrix(MatrixWrapper):
    # A = J @ M^{-1} @ J.T, but rather than storing A, we store M and J
    def __init__(self, M: MMatrix, J):
        self.M = M
        self.J = J

    def matvec(self, x):
        return self.J @ self.M.solve(self.J.T @ x)

    def dot(self, x):
        return self.matvec(x)

    def materialize(self):
        return self.J @ self.M.materialize_inverse() @ self.J.T


class HMatrix(MatrixWrapper):
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

    def materialize(self):
        return self.A.materialize() + self.E.toarray()
