import numpy as np
from . import TimeRangeType, RangeType, PointType
import trajectories as tj, pdb
from scipy.interpolate import interp1d

class Interpolation_(tj.Trajectory_):
    def __init__(self,
                 t_range: TimeRangeType = [0., 1.],
                 trajectory: np.ndarray = None,
                 axis: int = -2,
                 dim: int = None):
        assert trajectory is not None, "Interpolation_ needs trajectory keyword"
        if dim is not None:
            assert trajectory.shape[-1] in (1, dim), "trajectory shape %s do not match with dim %s"%(trajectory.shape[-1], dim)
            if trajectory.shape[-1] == 1:
                trajectory = np.repeat(trajectory, dim, axis=-1)
        self.trajectory = trajectory
        self.t_range = t_range
        self.axis = -2

    def __call__(self, t, *args, **kwargs):
        t_range = tj.expand([self.t_range[0], self.t_range[1]], None, t)
        trajectory = tj.expand(self.trajectory, None, t)
        t_interp = np.linspace(t_range[0], t_range[1], self.trajectory.shape[-2])[..., 0]
        interp = interp1d(t_interp, trajectory, axis=self.axis)
        trajectory = interp(t)
        return trajectory


