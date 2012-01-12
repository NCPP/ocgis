from base import OcgFunction
import numpy as np
import groups


class Mean(OcgFunction):
    description = 'Mean value for the series.'
    Group = groups.BasicStatistics
    
    @staticmethod
    def calculate(values):
        return(np.mean(values))
    
    
class Median(OcgFunction):
    description = 'Median value for the series.'
    Group = groups.BasicStatistics
    
    @staticmethod
    def calculate(values):
        return(np.median(values))
    
    
class Max(OcgFunction):
    description = 'Max value for the series.'
    Group = groups.BasicStatistics
    
    @staticmethod
    def calculate(values):
        return(max(values))
    
    
class Min(OcgFunction):
    description = 'Min value for the series.'
    Group = groups.BasicStatistics
    
    @staticmethod
    def calculate(values):
        return(min(values))
    
    
class StandardDeviation(OcgFunction):
    description = 'Standard deviation for the series.'
    name = 'std'
    test = 'Standard Deviation'
    Group = groups.BasicStatistics
    
    @staticmethod
    def calculate(values):
        return(np.std(values))
    
    
class Between(OcgFunction):
    description = 'Count of values between {0} and {1} (inclusive).'
    Group = groups.Thresholds
    
    @staticmethod
    def calculate(values,lower=None,upper=None):
        if lower is None or upper is None:
            raise(ValueError('a lower and upper limit are required'))
        days = filter(lambda x: x >= lower and x <= upper, values)
        return(len(days))
    
    
class GreaterThan(OcgFunction):
    text = 'Greater Than'
    name = 'gt'
    description = 'Count of values greater than {0} in the series (exclusive).'
    Group = groups.Thresholds
    
    @staticmethod
    def calculate(values,threshold=None):
        if threshold is None:
            raise(ValueError('a threshold must be passed'))
        days = filter(lambda x: x > threshold, values)
        return(len(days))
    
    
class LessThan(OcgFunction):
    text = 'Less Than'
    name = 'lt'
    description = 'Count of values less than {0} in the series (exclusive).'
    Group = groups.Thresholds
    
    @staticmethod
    def calculate(values,threshold=None):
        if threshold is None:
            raise(ValueError('a threshold must be passed'))
        days = filter(lambda x: x < threshold, values)
        return(len(days))