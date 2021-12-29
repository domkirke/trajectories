import trajectories as tj, numpy as np
from typing import Iterable, Union

class Uniform_(tj.Trajectory_):
    def __init__(self,
                 amp: Union[float, Iterable[float]] = 1.0,
                 dim: int = None,
                 fixed: bool = False,
                 centered: bool = True):
        self.amp = np.array(tj.checklist(amp))
        self.dim = dim
        self.fixed = fixed
        self.centered = centered

    def __repr__(self):
        return "Uniform_(%s, fixed=%s, centered=%s)"%(self.amp, self.fixed, self.centered)

    def __call__(self, t, *args, dim=None, **kwargs):
        dim = self.dim or dim or tj.GLOBAL_DIM
        amp = tj.expand(self.amp, dim)
        noise_shape = (*t.shape, *amp.shape) if not self.fixed else amp.shape
        noise = np.random.rand(*noise_shape)
        if self.centered:
            noise = noise * 2 - 1
        return amp * noise

class Normal_(tj.Trajectory_):
    def __init__(self,
                 mean: Union[float, Iterable[float]] = 0.0,
                 stddev: Union[float, Iterable[float]] = 1.0,
                 dim: int = None,
                 fixed: bool = False):
        self.mean = np.array(tj.checklist(mean))
        self.stddev = np.array(tj.checklist(stddev))
        self.dim = dim
        self.fixed = fixed

    def __repr__(self):
        return "Normal_(mean=%s, stddev=%s, fixed=%s)"%(self.mean, self.stddev, self.fixed)

    def __call__(self, t, *args, dim=None, **kwargs):
        dim = self.dim or dim or tj.GLOBAL_DIM
        mean = tj.expand(self.mean, dim)
        stddev = tj.expand(self.stddev, dim)
        noise_shape = (*t.shape, *mean.shape) if not self.fixed else mean.shape
        noise = stddev * np.random.randn(*noise_shape) + mean
        return noise

class RandomWalk_(Normal_):
    def __init__(self, *args, axis=-2, **kwargs):
        super(RandomWalk_, self).__init__(*args, **kwargs)
        self.axis = axis

    def __repr__(self):
        return "RandomWalk_(%s, axis=%s, fixed=%s, centered=%s)"%(self.stddev, self.axis, self.fixed, self.centered)

    def __call__(self, *args, **kwargs):
        out = super(RandomWalk_, self).__call__(*args, **kwargs)
        return np.cumsum(out, axis=self.axis)