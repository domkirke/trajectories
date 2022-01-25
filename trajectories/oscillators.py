import numpy as np
import trajectories as tj
from typing import Union, Tuple, Iterable

RangeType = Union[int, float, Iterable[int], Iterable[float]]


class Sawtooth_(tj.Trajectory_):
    def __init__(self,
                 freq: RangeType = 1.0,
                 phase: RangeType = 0.0,
                 amplitude: RangeType = 1.0,
                 origin = 0.0,
                 dim: int = None):
        self.freq = tj.check_param(freq)
        self.phase = tj.check_param(phase)
        self.amplitude = tj.check_param(amplitude)
        self.origin = tj.check_param(origin)
        self.dim = dim

    def __call__(self, t, dim=None, **kwargs):
        dim = dim or self.dim or tj.GLOBAL_DIM
        traj = np.zeros((*t.shape, dim))
        t = tj.expand(t, dim)
        bs = tj.get_batch_shape(t)
        # generate sawtooth
        freq = tj.expand(self.freq, dim, t)
        phase = tj.expand(self.phase, dim, t)
        amplitude = tj.expand(self.amplitude, dim, t)
        origin = tj.expand(self.origin, dim, t)[..., np.newaxis, :]
        for d in range(dim):
            t_offset = np.floor(np.amin(t) / freq[..., d]) - 2
            f_tmp = tj.expand_as(freq[..., d], t[..., d])
            p_tmp = tj.expand_as(phase[..., d], t[..., d])
            traj[..., d] = np.fmod(t[..., d] - tj.expand_as(t_offset, t[..., d])*f_tmp - p_tmp, f_tmp) / f_tmp
            # cetner
            traj[..., d] = (traj[..., d] * 2 - 1) * tj.expand_as(amplitude[..., d], t[..., d])
        traj = traj + origin
        return traj

class Square_(Sawtooth_):
    def __init__(self, pulse_width: RangeType = 0.5, **kwargs):
        super(Square_, self).__init__(**kwargs)
        self.pulse_width = tj.check_param(pulse_width)

    def __call__(self, t, dim=None, **kwargs):
        dim = dim or self.dim or tj.GLOBAL_DIM
        traj = np.zeros((*t.shape, dim))
        t = tj.expand(t, dim)
        bs = tj.get_batch_shape(t)
        # generate sawtooth
        freq = tj.expand(self.freq, dim, t)
        phase = tj.expand(self.phase, dim, t)
        amplitude = tj.expand(self.amplitude, dim, t)
        pulse_width = tj.expand(self.pulse_width, dim, t)
        origin = tj.expand(self.origin, dim, t)[..., np.newaxis, :]
        for d in range(dim):
            t_offset = np.floor(np.amin(t) / freq[..., d]) - 2
            f_tmp = tj.expand_as(freq[..., d], t[..., d])
            p_tmp = tj.expand_as(phase[..., d], t[..., d])
            traj[..., d] = np.fmod(t[..., d] - tj.expand_as(t_offset, t[..., d])*f_tmp - p_tmp, f_tmp) / f_tmp
            # cetner
            traj[..., d] = traj[..., d] > tj.expand_as(pulse_width[..., d], t[..., d])
            traj[..., d] = (traj[..., d] * 2 - 1) * tj.expand_as(amplitude[..., d], t[..., d])
        traj = traj + origin
        return traj


class Sin_(tj.Trajectory_):
    def __init__(self,
                 freq: RangeType = 1.0,
                 phase: RangeType = 0.0,
                 amplitude: RangeType = 1.0,
                 origin = 0.0,
                 dim: int = None):
        self.freq = tj.check_param(freq)
        self.phase = tj.check_param(phase)
        self.amplitude = tj.check_param(amplitude)
        self.origin = tj.check_param(origin)
        self.dim = dim

    def __call__(self, t, dim=None, **kwargs):
        dim = dim or self.dim or tj.GLOBAL_DIM
        traj = np.zeros((*t.shape, dim))
        t = tj.expand(t, dim)
        bs = tj.get_batch_shape(t)
        # generate sine
        freq = tj.expand(self.freq, dim, t)
        phase = tj.expand(self.phase, dim, t)
        amplitude = tj.expand(self.amplitude, dim, t)
        origin = tj.expand(self.origin, dim, t)[..., np.newaxis, :]
        for d in range(dim):
            f_tmp = tj.expand_as(freq[..., d], t[..., d])
            p_tmp = tj.expand_as(phase[..., d], t[..., d])
            a_tmp = tj.expand_as(amplitude[..., d], t[..., d])
            traj[..., d] = a_tmp * np.sin(2 * np.pi * (f_tmp * t[..., d] + p_tmp))
            # cetner
        traj = traj + origin
        return traj
