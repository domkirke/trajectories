import numpy as np, abc, sys
from typing import Union, List
sys.path.append('../')
import trajectories as traj


# implements various points that can be called with optional arguments.
class Point(object):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        super(Point, self).__init__()
        self.dim = kwargs.get('dim')


class Origin(Point):
    def __repr__(self):
        return "Origin"
    def __call__(self, *args, **kwargs):
        dim = self.dim or kwargs.get('dim')
        return np.zeros([dim])


class Uniform(Point):
    def __init__(self, range=np.array([-1, 1]), **kwargs):
        super(Uniform, self).__init__(**kwargs)
        self.range = range

    def __repr__(self):
        return "Uniform(%s)"%self.range

    def __call__(self,  *args, dim=None, **kwargs):
        range = self.range
        dim = dim or self.dim or traj.GLOBAL_DIM
        noise = np.random.rand(dim)
        noise = noise*(range[..., 1]-range[..., 0]) + range[..., 0]
        return noise


class Normal(Point):
    def __init__(self, mean=np.array([0]), stddev=np.array([1.]), **kwargs):
        super(Normal, self).__init__(**kwargs)
        self.mean = mean
        self.stddev = stddev

    def __repr__(self):
        return "Normal(%s,%s)"%(self.mean, self.stddev)

    def __call__(self,  *args, dim=None, **kwargs):
        dim = dim or self.dim or traj.GLOBAL_DIM
        mean = traj.expand(self.mean, dim)
        var = traj.expand(self.stddev, dim) ** 2
        noise = np.random.randn(dim) * var + mean
        return noise
