import madrona_stick as s

from reduced_primal import reduced_primal

if __name__ == "__main__":
    num_worlds = 1
    app = s.HeadlessRun(num_worlds, 40)
    app.run(reduced_primal)
