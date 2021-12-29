import numpy as np, pdb
from scipy.interpolate import interp1d, griddata
import trajectories as traj
from . import  TimeRangeType, RangeType
from itertools import product
from .trajectory import Trajectory_
from .misc import euc2sph, sph2euc
from .checkups import parsepoint, expand, get_batch_shape, expand_as, check_param
from typing import Union, Tuple, List
from .points import Point



def line_generator(t, t_range, y_range, polar=False, dim=None, out_mode="unfold"):
    y_origin, y_end = expand(y_range, dim, t)
    if polar:
        y_origin = euc2sph(y_origin)
        y_end = euc2sph(y_end)
    t_origin, t_end = expand(t_range, dim, t)
    trajs = []
    for i in range(dim):
        a = (y_end[..., i] - y_origin[..., i]) / (t_end[..., i] - t_origin[..., i])
        b = y_origin[..., i] - a * t_origin[..., i]
        out = expand_as(a, t[..., i])  * t[..., i] + expand_as(b, t[..., i])
        if out_mode == "clamp":
            if len(t.shape) <= 2:
                out[t[..., i] < t_origin[..., i, np.newaxis]] = y_origin[..., i]
                out[t[..., i] > t_end[..., i]] = y_end[..., i]
            else:
                for s in product(*[list(range(b)) for b in t.shape[:-2]]):
                    out.__setitem__((*s, t.__getitem__((*s, ..., i)) > t_end.__getitem__((*s, ..., i))),
                                    y_end.__getitem__((*s, i)))
                    out.__setitem__((*s, t.__getitem__((*s, ..., i)) < t_origin.__getitem__((*s, ..., i))),
                                    y_origin.__getitem__((*s, i)))

        trajs.append(out)
    out = np.stack(trajs, axis=-1)
    if polar:
        out = sph2euc(out)
    return out

class Line_(Trajectory_):
    def __init__(self,
                 t_range: TimeRangeType = [0., 1.],
                 y_range: RangeType = [0., 1.],
                 dim: int = None,
                 out_mode: str = "unfold"):
        self.t_range = [check_param(t_range[0]), check_param(t_range[1])]
        self.y_range = [parsepoint(y_range[0], dim=dim), parsepoint(y_range[1], dim=dim)]
        self.out_mode = out_mode
        self.dim = dim

    def __repr__(self):
        return "Line(%s,%s)"%(self.t_range, self.y_range)

    def __call__(self, t, *args, dim=None, **kwargs):
        dim = dim or self.dim or traj.GLOBAL_DIM
        t = expand(t, dim)
        return line_generator(t, t_range = self.t_range, y_range = self.y_range, dim=dim, out_mode=self.out_mode)


class PolarLine_(Line_):
    def __call__(self, t, *args, dim=None, **kwargs):
        dim = dim or self.dim or traj.GLOBAL_DIM
        t = expand(t, dim)
        return line_generator(t, t_range = self.t_range, y_range = self.y_range, dim=dim, out_mode=self.out_mode, polar=True)






