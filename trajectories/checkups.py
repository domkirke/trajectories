import numpy as np
from numbers import Real
import sys; sys.path.append('../')
import trajectories
from trajectories import DimException, GLOBAL_DIM, Trajectory_, checklist
from .points import Point

def get_batch_shape(phase, with_phase=True):
    return phase.shape[:-(1 + int(with_phase))]

def parsepoint(point, dim=None):
    """if the point is a class, init the point"""
    if isinstance(point, type):
        point = point(dim=dim)
    else:
        point = check_param(point)
    return point

def expand_as(arr, target, dir="right"):
    missing_dims = len(target.shape) - len(arr.shape)
    arr_shape = len(arr.shape)
    if missing_dims < 0:
        raise ValueError("expand_as tensor bigger than target : %s, target %s "%(arr.shape, target.shape))
    for i in range(missing_dims):
        if dir == "right":
            arr = arr[..., np.newaxis]
            arr = arr.repeat(target.shape[arr_shape + i], -1)
        elif dir == "left":
            arr = arr[np.newaxis]
            arr = arr.repeat(target.shape[-(i + 1 + missing_dims)], 0)
    return arr


def expand(t, dim=None, phase=None, with_phase=False):
    batch = None
    if phase is not None:
        batch = phase.shape[:-(1 + int(not with_phase))]
    if isinstance(t, (list, tuple)):
        return type(t)([expand(t_tmp, dim=dim, phase=phase, with_phase=with_phase) for t_tmp in t])
    if isinstance(t, Trajectory_):
        if t is None:
            raise ValueError("if a Trajectory_ is given as a parameter, has to be called with phase")
        t = t(phase)
    if isinstance(t, Point):
        t = t(dim=dim)
    elif not isinstance(t, np.ndarray):
        t = np.array([t])
    # check dimensionality
    if dim is not None:
        if t.shape[-1] == 1:
            t = t.repeat(dim, axis=-1)
        elif t.shape[-1] != dim:
            t = t[..., np.newaxis]
            t = t.repeat(dim, axis=-1)
    # check batch_shape
    if batch is not None:
        if t.shape[:-1] != batch:
            missing_dims = len(batch) - len(t.shape[:-1])
            t = t.__getitem__( (np.newaxis,)*missing_dims)
            for m in range(missing_dims):
                t = t.repeat(batch[m], m)
    return t

def check_param(arr):
    if isinstance(arr, (np.ndarray, Trajectory_)):
        return arr
    else:
        return np.array(checklist(arr))
