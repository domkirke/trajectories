from typing import Union, Tuple, List, Iterable
FS = None
GLOBAL_DIM = None
TimeRangeType = Tuple[Union[float, Iterable[float]], Union[float, Iterable[float]]]

class DimException(Exception):
    pass

from .misc import *
from .trajectory import Trajectory_
from .checkups import *
from .points import *
PointType = Union[int, float, Point, np.ndarray]
RangeType = Tuple[PointType, PointType]

from .linear import Line_, PolarLine_
from .random import Uniform_, Normal_, RandomWalk_
from .circular import Circle_, Helix_, Ellipse_
from .oscillators import Sawtooth_, Square_, Sin_
#from .circular import Circle, circle_generator, Ellipse, ellipse_generator
#from .oscillators import *


