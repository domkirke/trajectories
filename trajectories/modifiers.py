import trajectories as traj, numpy as np

class TrajectoryEffect(traj.Trajectory_):
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def transform(self, trajectory):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.transform(self.trajectory(*args, **kwargs))

class Scramble(TrajectoryEffect):
    def __init__(self, trajectory):
        super(Scramble, self).__init__(trajectory)
        self.scramble_map = None
        if trajectory.dim is not None:
            self._init_scramble_map(trajectory.dim)

    def __repr__(self):
        return "Scramble(%s)"%self.trajectory

    @property
    def initialized(self):
        return self.scramble_map is not None

    def _init_scramble_map(self, dim):
        self.dim = dim
        self.scramble_map = np.random.permutation(self.dim)

    def transform(self, trajectory):
        if not self.initialized:
            self._init_scramble_map(trajectory.shape[-1])
        return trajectory[..., self.scramble_map]


