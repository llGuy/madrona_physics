import numpy as np
import cvxpy as cp
import madrona_stick as s

def cvx_solve(A, v0):
    print("in python!")
    print(A.shape)
    print(v0.shape)

num_worlds = 1
app = s.PhysicsApp(num_worlds)
app.run(cvx_solve)
