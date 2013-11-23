import math
import statistics
import thresholds
from ocgis.calc.library.index import dynamic_kernel_percentile, heat_index, duration


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
    
    @classmethod
    def append(cls,value):
        cls.reg.append(value)