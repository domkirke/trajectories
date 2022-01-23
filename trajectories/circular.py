import numpy as np
from typing import Union, Tuple
from . import TimeRangeType, RangeType, PointType
import trajectories as tj, pdb
from trajectories.linear import line_generator
from scipy.stats import ortho_group
from scipy.linalg import norm, inv


class Circle_(tj.Trajectory_):
    def __init__(self,
                 t_range: TimeRangeType = [0., 1.],
                 theta_range: RangeType = [0., 1.],
                 radius: RangeType = [1.],
                 center: PointType = [0.],
                 dim: int = None):
        self.t_range = t_range
        self.theta_range = theta_range
        self.radius = np.array(radius)
        self.center = tj.parsepoint(center)
        self.dim = dim

    def get_angles(self, t, t_range, theta_range):
        theta_range = 2 * np.pi * np.array(theta_range)
        a = (theta_range[1] - theta_range[0]) / (t_range[1] - t_range[0])
        b = theta_range[0] - a * t_range[0]
        if tj.get_batch_shape(t) != tuple():
            a = a[..., np.newaxis, :]
            b = b[..., np.newaxis, :]
        return a * t + b

    def __call__(self, t, *args, dim=None, **kwargs):
        dim = dim or self.dim or tj.GLOBAL_DIM
        if dim == 1:
            raise tj.DimException('Circle_ cannot be called with dim=1')
        # generate circle
        t = tj.expand(t, 1)
        t_range = tj.expand([self.t_range[0], self.t_range[1]], None, t)
        theta_range = tj.expand([self.theta_range[0], self.theta_range[1]], None, t)
        angles = self.get_angles(t, t_range, theta_range)
        radius = tj.expand(self.radius, 2, t)
        circ_traj = np.concatenate([tj.expand_as(radius[..., 0], angles) * np.cos(angles),
                                    tj.expand_as(radius[..., 1], angles) * np.sin(angles)], axis = -1)
        # embed circle
        if dim > 2:
            circ_traj_inclusion = np.concatenate([circ_traj, np.zeros((*circ_traj.shape[:-1], dim - circ_traj.shape[-1]))], axis=-1)
            orth_mat = ortho_group.rvs(dim)
            circ_traj = np.matmul(circ_traj_inclusion, orth_mat)
        origin = tj.parsepoint(self.center, dim=dim)
        circ_traj += origin
        return circ_traj

class Helix_(tj.Trajectory_):
    def __init__(self,
                 t_range: TimeRangeType = [0., 1.],
                 z_range: RangeType = [0., 1.],
                 n_spires: int = 3,
                 radius: Union[float, Tuple[float, float]] = 1.0,
                 dim: int = None,
                 random_rotate: bool = True):
        self.t_range = t_range
        self.z_range = z_range
        self.n_spires = int(n_spires)
        self.radius = np.array(tj.checklist(radius))
        self.dim = dim
        self.random_rotate = random_rotate

    def get_angles(self, t, t_range, theta_range):
        theta_range = 2 * np.pi * np.array(theta_range)
        a = (theta_range[1] - theta_range[0]) / (t_range[1] - t_range[0])
        b = theta_range[0] - a * t_range[0]
        if tj.get_batch_shape(t) != tuple():
            a = a[..., np.newaxis, :]
            b = b[..., np.newaxis, :]
        return a * t + b

    def get_z(self, t, t_range, z_range):
        a = (z_range[1] - z_range[0]) / (t_range[1] - t_range[0])
        b = z_range[0] - a * t_range[0]
        if tj.get_batch_shape(t) != tuple():
            a = a[..., np.newaxis, :]
            b = b[..., np.newaxis, :]
        return a * t + b

    def __call__(self, t, dim=None, **kwargs):
        dim = dim or self.dim or tj.GLOBAL_DIM
        if dim < 3:
            raise tj.DimException('Helix_ cannot be called with dim<3')
        # generate circle
        t = tj.expand(t, 1)
        t_range = tj.expand(self.t_range, None, t)
        z_range = tj.expand(self.z_range, None, t)
        theta_range = tj.expand([0., float(self.n_spires)], 1, t)
        angles = self.get_angles(t, t_range, theta_range)
        z = self.get_z(t, t_range, z_range)
        radius = tj.expand(self.radius, 2, t)
        helix_traj = np.concatenate([tj.expand_as(radius[..., 0], angles) * np.cos(angles),
                               tj.expand_as(radius[..., 1], angles) * np.sin(angles),
                               z], axis=-1)
        # embed circle
        if dim > 3:
            helix_traj = np.concatenate(
                [helix_traj, np.zeros((*helix_traj.shape[:-1], dim - helix_traj.shape[-1]))], axis=-1)
        if self.random_rotate:
            rot_mat = ortho_group.rvs(dim)
            batch_shape = helix_traj.shape[:-1]
            rot_mat = np.reshape(rot_mat, (*(1,)*len(batch_shape), *rot_mat.shape[-2:],))
            for i,b in enumerate(batch_shape):
                rot_mat = rot_mat.repeat(b, i)
            helix_traj = np.matmul(rot_mat, helix_traj[..., np.newaxis])[..., 0]
        return helix_traj


class Ellipse_(tj.Trajectory_):
    def __init__(self,
                 t_range: TimeRangeType = [0., 1.],
                 theta_range: RangeType = [0., 1.],
                 radius: RangeType = [1.],
                 phase: RangeType = [0.],
                 center: PointType = [0.],
                 dim: int = None):
        self.t_range = t_range
        self.theta_range = [tj.check_param(theta_range[0]), tj.check_param(theta_range[1])]
        self.radius = np.array(radius)
        self.phase = np.array(phase)
        self.center = tj.parsepoint(center)
        self.dim = dim

    def get_angles(self, t, t_range, theta_range, dim):
        out_ranges = []
        for d in range(dim-1):
            if d < dim - 2:
                to = theta_range[0][..., d] * 2 * np.pi
                te = theta_range[1][..., d] * 2 * np.pi
            else:
                to = theta_range[0][..., d] * 2 * np.pi
                te = theta_range[1][..., d] * 2 * np.pi
            a = (te - to) / (t_range[1][..., d] - t_range[0][..., d])
            b = to - a * t_range[0][..., d]
            if tj.get_batch_shape(t) != tuple():
                a = tj.expand_as(a, t[..., d])
                b = tj.expand_as(b, t[..., d])
            out_ranges.append(a * t[..., d] + b)
        return np.stack(out_ranges, -1)

    def __call__(self, t, *args, dim=None, **kwargs):
        dim = dim or self.dim or tj.GLOBAL_DIM
        if dim == 1:
            raise tj.DimException('Circle_ cannot be called with dim=1')
        # generate circle
        t = tj.expand(t, dim)
        t_range = tj.expand(self.t_range, dim, t)
        theta_range = tj.expand(self.theta_range, dim, t)
        phase = tj.expand(self.phase, dim - 1, t)
        if phase.shape[-1] != dim - 1:
            raise Exception('phase shape not valid : %s, should be %s'%(phase.shape[-1], dim - 1))
        angles = self.get_angles(t, t_range, theta_range, dim=dim)
        angles = angles + (2 * np.pi * phase[..., np.newaxis, :])
        radius = self.radius
        if radius.shape < t.shape:
            radius = radius[..., np.newaxis, :]
        radius = tj.expand(radius, 1, t, with_phase=True)
        if radius.shape[-2] != angles.shape[-2]:
            radius = np.repeat(radius, angles.shape[-2], axis=-2)
        sph_coo = np.concatenate([radius, angles], axis=-1)
        euc_coo = tj.sph2euc(sph_coo)
        # embed circle
        origin = tj.expand(self.center, dim, t, with_phase=True)
        euc_coo += origin
        return euc_coo
