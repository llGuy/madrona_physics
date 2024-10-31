import numpy as np
import cvxpy as cp
import madrona_stick as s

def cvx_solve(A, v0, result):
    print(A)
    print(v0)

    result += np.array([42, 56, 2])

num_worlds = 1
app = s.PhysicsApp(num_worlds)
app.run(cvx_solve)
