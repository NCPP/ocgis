import math
import statistics
import thresholds
from ocgis.calc.library.index import dynamic_kernel_percentile, heat_index, duration
from ocgis.util.helpers import itersubclasses


class FunctionRegistry(dict):
    reg = []
    
    def __init__(self):
        super(FunctionRegistry,self).__init__()
        
        self.reg += [math.Divide,math.NaturalLogarithm]
        self.reg += [statistics.FrequencyPercentile,statistics.Mean,statistics.StandardDeviation,
                     statistics.Max,statistics.Median,statistics.Min]
        self.reg += [thresholds.Between,thresholds.Threshold]
        self.reg += [dynamic_kernel_percentile.DynamicDailyKernelPercentileThreshold,
                     heat_index.HeatIndex,duration.Duration,duration.FrequencyDuration]
        
        for cc in self.reg:
            self.update({cc.key:cc})
            
    def add_function(self,value):
        self.update({value.key:value})
    
    @classmethod
    def append(cls,value):
        cls.reg.append(value)


def register_icclim(function_registry):
    '''Register ICCLIM indices.
    
    :param function_registry: The target :class:`FunctionRegistry` object to hold
     ICCLIM index references.
    :type function_registry: :class:`FunctionRegistery`
    '''
    
    from ocgis.contrib import library_icclim
    for subclass in itersubclasses(library_icclim.AbstractIcclimFunction):
        function_registry.add_function(subclass)