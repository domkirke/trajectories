import numpy as np
import random
from .points import Point


def checklist(arr, n=1):
    if isinstance(arr, list):
        return arr
    else:
        return [arr]*n


def scale_traj(traj, old_min, old_max, new_min, new_max):
    return (traj - old_min)/(old_max - old_min)*(new_max-new_min)+new_min


def euc2sph(arr):
    arr_sph = np.zeros_like(arr, dtype=np.float)
    dim = arr.shape[-1]
    if dim == 2:
        arr_sph[..., 0] = np.sqrt((arr ** 2).sum(-1))
        arr_sph[..., 1] = np.arctan2(arr[..., 1], arr[..., 0])
    else:
        for d in range(dim):
            if d == 0:
                arr_sph[..., 0] = np.sqrt((arr ** 2).sum(-1))
            elif d < dim - 1:
                arr_sph[..., d] = np.arccos(arr[..., d-1] /
                                            np.sqrt((arr[..., (d-1):] ** 2).sum(-1)))
            else:
                div_term = np.sqrt(arr[..., -1] ** 2 + arr[..., -2] ** 2)
                arr_sph[arr[..., d] >= 0, d] = np.arccos(arr[arr[..., d]>=0, d-1] / div_term)
                arr_sph[arr[..., d] < 0, d] = 2*np.pi - np.arccos(arr[arr[..., d] < 0, d-1] / div_term)
    return arr_sph

def sph2euc(arr):
    arr_euc = np.zeros_like(arr, dtype=np.float)
    dim = arr.shape[-1]
    if dim == 2:
        arr_euc[..., 0] = arr[..., 0] * np.cos(arr[..., 1])
        arr_euc[..., 1] = arr[..., 0] * np.sin(arr[..., 1])
    else:
        for d in range(dim):
            if d == 0:
                arr_euc[..., 0] = arr[..., 0] * np.cos(arr[..., 1])
            elif d < dim - 1:
                arr_euc[..., d] = arr[..., 0] * np.prod(np.sin(arr[..., 1:(d+1)]), axis=-1) * np.cos(arr[..., d+1])
            else:
                arr_euc[..., d] = arr[..., 0] * np.prod(np.sin(arr[..., 1:]), axis=-1)
    return arr_euc


def uniform(n=1, r=[0, 1]):
    return np.array([random.random() * (r[1] - r[0]) + r[0] for _ in range(n)])

def normal(n=1, m=0.0, s=1.0):
    return np.random.normal(m, s, n)

