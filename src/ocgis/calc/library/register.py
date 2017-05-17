from ocgis.calc.library.index import heat_index, duration
from ocgis.calc.library.math import Convolve1D
from ocgis.util.helpers import itersubclasses
from . import math
from . import statistics
from . import thresholds


class FunctionRegistry(dict):
    reg = []

    def __init__(self):
        super(FunctionRegistry, self).__init__()

        self.reg += [math.Divide, math.NaturalLogarithm, math.Sum]
        self.reg += [statistics.FrequencyPercentile, statistics.Mean, statistics.StandardDeviation, statistics.Max,
                     statistics.Median, statistics.Min, Convolve1D, statistics.MovingWindow, statistics.DailyPercentile]
        self.reg += [thresholds.Between, thresholds.Threshold]
        self.reg += [heat_index.HeatIndex, duration.Duration]

        for cc in self.reg:
            self.update({cc.key: cc})

    def add_function(self, value):
        self.update({value.key: value})

    @classmethod
    def append(cls, value):
        cls.reg.append(value)


def register_icclim(function_registry):
    """
    Register ICCLIM indices.

    :param function_registry: The target :class:`FunctionRegistry` object to hold ICCLIM index references.
    :type function_registry: :class:`FunctionRegistery`
    """

    from ocgis.contrib import library_icclim
    for subclass in itersubclasses(library_icclim.AbstractIcclimFunction):
        function_registry.add_function(subclass)
