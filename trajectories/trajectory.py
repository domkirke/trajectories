import numpy as np, abc, math
from numpy import ceil, floor, round
from trajectories import checklist
ROUND_STRATEGY = round



# dummy class that is used to ignore a given argument in trajectory classes
class Ignore(object):
    pass

class Trajectory_(object):
    pass

    def __add__(self, other):
        if isinstance(other, TrajectoryAlgebra_):
            return TrajectoryAlgebra_([self]+other.trajectories,
                                      weight=[1.0] + other.weight)
        elif isinstance(other, Trajectory_):
            return TrajectoryAlgebra_([self, other], weight=[1.0, 1.0])
        else:
            raise TypeError

    def __mul__(self, other):
        try:
            other = float(other)
        except:
            raise TypeError("Trajectory_ objects only be multiplied with float-like objects")
        return TrajectoryAlgebra_([self], [other])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __and__(self, other):
        if isinstance(other, TrajectoryCompound_):
            return TrajectoryCompound_([self]+other.trajectories)
        elif isinstance(other, (Trajectory_, TrajectoryAlgebra_)):
            return TrajectoryCompound_([self] + [other])


class TrajectoryAlgebra_(object):
    def __init__(self, trajectories, weight=1.0):
        self.trajectories = checklist(trajectories)
        self.weight = checklist(weight, n=len(self.trajectories))

    @property
    def dim(self):
        dims = [t.dim for t in self.trajectories]
        if None in dims:
            return None
        else:
            assert len(set(dims)) == 1, "dims in %s are not consistent"%self
        return dims[0]

    def __and__(self, other):
        if isinstance(other, TrajectoryCompound_):
            return TrajectoryCompound_([self]+other.trajectories)
        elif isinstance(other, (Trajectory_, TrajectoryAlgebra_)):
            return TrajectoryCompound_([self] + [other])

    def __repr__(self):
        repr = "("
        for i, t in enumerate(self.trajectories):
            w = self.weight[i]
            if i == 0:
                if w == 1.0:
                    repr += t.__repr__()
                else:
                    repr += str(w) + "*" + t.__repr__()
            else:
                w_str = str(abs(w)) + " * " if abs(w) != 1.0 else ""
                if w < 0:
                    repr += "\n - " + w_str + t.__repr__()
                else:
                    repr += "\n + " + w_str + t.__repr__()
        repr += ")"
        return repr

    def __call__(self, *args, **kwargs):
        buff = None
        for i, t in enumerate(self.trajectories):
            out = self.weight[i] * t(*args, **kwargs)
            if buff is None:
                buff = out
            else:
                buff += out
        return buff

    def insert(self, idx, t, weight=1.0):
        self.trajectories.insert(idx, t)
        self.weight.insert(idx, weight)


class TrajectoryCompound_(Trajectory_):
    def __init__(self, trajectories):
        self.trajectories = checklist(trajectories)

    @property
    def dim(self):
        dims = []
        for t in self.trajectories:
            dims.append(t.dim)
        if None in dims:
            return None
        return sum(dims)

    def __repr__(self):
        repr = ""
        for i, t in enumerate(self.trajectories):
            if i == 0:
                repr += "(" + str(t)
            else:
                repr += "\n & " + str(t)
        repr += ")"
        return repr

    def __and__(self, other):
        if isinstance(other, TrajectoryCompound_):
            return TrajectoryCompound_(self.trajectories + other.trajectories)
        elif isinstance(other, (Trajectory_, TrajectoryAlgebra_)):
            return TrajectoryCompound_(self.trajectories + [other])
        raise TypeError()

    def __rand__(self, other):
        if isinstance(other, TrajectoryCompound_):
            return TrajectoryCompound_(other.trajectories + self.trajectories)
        elif isinstance(other, Trajectory_):
            return TrajectoryCompound_([other] + self.trajectories)
        else:
            raise TypeError()

    def __call__(self, *args, **kwargs):
        return np.concatenate([t(*args, **kwargs) for t in self.trajectories], axis=-1)

