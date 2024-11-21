"""
friction cone related stuff
"""
import numpy as np

class FrictionCones:
    def __init__(self, mus):
        self.cones = []
        for mu in mus:
            self.cones.append(Cone(mu))

    def get_min_t(self, x, d):
        min_t = np.inf
        for i, cone in enumerate(self.cones):
            t = cone.line_intersect(x[3 * i: 3 * i + 3], d[3 * i: 3 * i + 3])
            min_t = min(min_t, t)
        return min_t

    def in_cone(self, x):
        for i, cone in enumerate(self.cones):
            if not cone.in_cone(x[3 * i: 3 * i + 3]):
                return False
        return True



class Cone:
    def __init__(self, mu):
        self.mu = mu

    def line_intersect(self, p, d):
        """
        returns the intersection time t of line p + td with the cone
        """
        a = self.mu ** 2 * d[0] ** 2 - d[1] ** 2 - d[2] ** 2
        b = 2 * self.mu ** 2 * p[0] * d[0] - 2 * p[1] * d[1] - 2 * p[2] * d[2]
        c = self.mu ** 2 * p[0] ** 2 - p[1] ** 2 - p[2] ** 2
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            assert False
        t1 = (-b + np.sqrt(disc)) / (2 * a)
        t2 = (-b - np.sqrt(disc)) / (2 * a)
        print(t1, t2)
        assert t1 >= 0 or t2 >= 0

        # first positive t
        if t2 >= 0:
            return t2
        return t1


    def in_cone(self, x):
        return self.mu * x[0] >= np.linalg.norm(x[1:])

