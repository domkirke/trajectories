import trajectories
import numpy as np
import matplotlib.pyplot as plt

# trajectories is a simple package providing methods for trajectory generation.

# example of line generation
from trajectories import Origin, Uniform, Normal, Line
dim = 3
origin = Origin
end = Uniform(dim=dim, scale=[-1, 1])
line = Line(n_steps=20, dim=dim, origin=origin, end=end)
traj = line()

fig, ax = plt.subplots(3, 1)
for i in range(3):
    ax[i].plot(traj[:,i])
plt.show()
