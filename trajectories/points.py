import numpy as np, abc
from .trajectory import Trajectory
from .misc import Ignore

# implements various points that can be called with optional arguments.
class Point(Trajectory):
    @abc.abstractmethod
    def get_callback(self):
        raise AttributeError('Point objects do not have any callback')

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        kwargs['dim'] = kwargs.get('dim')
        super(Point, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        init_args = self.call_args
        init_kwargs = dict(self.kwargs)
        for i in range(len(args)):
            if not issubclass(type(args[i]), Ignore):
                init_args[i] = args[i]
        for k, v in kwargs.items():
            init_kwargs[k] = v
        return None


class Origin(Point):
    def __call__(self, *args, **kwargs):
        dim = kwargs.get('dim') or self.kwargs.get('dim') or 1
        return np.zeros([dim])


class Uniform(Point):
    def __call__(self,  *args, **kwargs):
        dim = kwargs.get('dim') or self.call_kwargs.get('dim') or 1
        range = kwargs.get('range') or self.call_kwargs.get('range') or [-1, 1]

        range = np.array(range)
        if len(range.shape) == 1:
            range = np.repeat(range[np.newaxis], dim, 0)
        else:
            assert range.shape[0] == dim, "range must be a list of two elements or a numpy array of dims [dim, 2]"

        noise = np.random.rand(dim)
        noise = noise*(range[:, 1]-range[:, 0]) + range[:, 0]
        return noise


class Normal(Point):
    def __call__(self,  *args, **kwargs):
        dim = kwargs.get('dim') or self.kwargs.get('dim') or 1
        mean = kwargs.get('mean') or self.kwargs.get('mean') or 0
        cov =  kwargs.get('scale') or self.kwargs.get('scale') or [1.]

        mean = np.array(mean)
        if len(mean.shape) == 1:
            if range.shape[0] == 1:
                mean = np.repeat(mean[np.newaxis], dim, 1)
            else:
                assert mean.shape[0] == dim, "mean must be a list of two elements or a numpy array of dims [dim, 2]"

        cov = np.array(cov)
        if len(cov.shape) == 1:
            if cov.shape[0] == 1:
                cov = np.eye(dim) * cov[0]
            else:
                assert cov.shape[0] == dim, "if the given scale is a 1-d vector of more than one element, its dimension"\
                                             " should match the given dimension"
        elif len(cov.shape) == 2:
            assert cov.shape[0] == cov.shape[1] == dim, "if the given scale is a 2-d vector, it has to be a "\
                                                        "(%s,%s) squared matrix"%(dim, dim)

        noise = np.random.multivariate_normal(mean=mean, cov=cov)
        return noise


