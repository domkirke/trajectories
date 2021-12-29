import trajectories as traj
import numpy as np


class Zero_(traj.Trajectory_):
    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def __call__(self, t, dim=None, *args, **kwargs):
        dim = dim or self.dim or traj.GLOBAL_DIM
        t = traj.expand(t, dim)
        return np.zeros_like(t)