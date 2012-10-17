from ocgis.calc.wrap import groups
from ocgis.calc.wrap.base import OcgFunction, OcgCvArgFunction, OcgArgFunction
import numpy as np
from ocgis.util.helpers import iter_array


class SampleSize(OcgFunction):
    description = 'Statistical sample size.'
    Group = groups.BasicStatistics
    name = 'n'
    dtype = int
    
    @staticmethod
    def _calculate_(values,axis):
        ret = np.sum(~values.mask,axis=axis)
#        if axis is 0:
#            ret = np.ma.array(ret,mask=values.mask)
        return(ret)


class Median(OcgFunction):
    description = 'Median value for the series.'
    Group = groups.BasicStatistics
    dtype = float
    
    @staticmethod
    def _calculate_(values,axis):
        return(np.median(values,axis=axis))
    
    
class Mean(OcgFunction):
    description = 'Mean value for the series.'
    Group = groups.BasicStatistics
    dtype = float
    
    @staticmethod
    def _calculate_(values,axis):
        return(np.mean(values,axis=axis))
    
    
class Max(OcgFunction):
    description = 'Max value for the series.'
    Group = groups.BasicStatistics
    dtype = float
    
    @staticmethod
    def _calculate_(values,axis):
        return(np.max(values,axis=axis))
    
    
class Min(OcgFunction):
    description = 'Min value for the series.'
    Group = groups.BasicStatistics
    dtype = float
    
    @staticmethod
    def _calculate_(values,axis):
        return(np.min(values,axis=axis))
    
    
class StandardDeviation(OcgFunction):
    description = 'Standard deviation for the series.'
    Group = groups.BasicStatistics
    dtype = float
    name = 'std'
    
    @staticmethod
    def _calculate_(values,axis):
        return(np.std(values,axis=axis))
    
    
class MaxConsecutive(OcgArgFunction):
    name = 'max_cons'
    nargs = 2
    Group = groups.Thresholds
    dtype = int
    description = ('Maximum number of consecutive occurrences in the sequence'
                   ' where the logical operation returns TRUE.')
    
    @staticmethod
    def _calculate_(values,axis,threshold=None,operation=None):
        ## time index reference
        ref = np.arange(0,values.shape[0])
        ## storage array for counts
        store = np.empty(list(values.shape)[1:])
#        store = np.ma.array(store,mask=values.mask[0,0,:].reshape(store.shape))
        ## perform requested logical operation
        if operation == 'gt':
            arr = values > threshold
        elif operation == 'lt':
            arr = values < threshold
        elif operation == 'gte':
            arr = values >= threshold
        elif operation == 'lte':
            arr = values <= threshold

#        ## index iterator
#        it = itertools.product(*[range(0,values.shape[ii]) \
#                                 for ii in range(2,4)])
        ## find longest sequence for each geometry across time dimension
        for xidx,yidx in iter_array(values[0,:]):
            vec = arr[:,xidx,yidx]
            ## collapse data if no axis provided
            if axis is None:
                vec = vec.reshape(-1)
            ## check first if there is a longer series than 1
            if np.any(np.diff(ref[vec]) == 1):
                split_idx = ref[np.diff(vec)] + 1
                splits = np.array_split(vec,split_idx)
                sums = [a.sum() for a in splits if np.all(a)]
                fill = np.max(sums)
            ## case of only a singular occurrence
            elif np.any(vec):
                fill = 1
            ## case for no occurrence
            else:
                fill = 0
            store[xidx,yidx] = fill
        
        ## summarize across geometries if axis is collapsed
        if axis is None:
            store = np.max(store)
            
        return(store)
        

class Between(OcgArgFunction):
    nargs = 2
    description = 'Count of values falling within the limits lower and upper (inclusive).'
    Group = groups.Thresholds
    dtype = int
    
    @staticmethod
    def _calculate_(values,axis,lower=None,upper=None):
        idx = (values >= lower)*(values <= upper)
        return(np.sum(idx,axis=axis))


class FooMulti(OcgCvArgFunction):
    description = 'Meaningless test statistic.'
    Group = groups.MultivariateStatistics
    dtype = float
    nargs = 2
    keys = ['foo','foo2']
    
    @staticmethod
    def _calculate_(foo=None,foo2=None):
        ret = foo + foo2
        ret = 2*ret
        ret = np.mean(ret,axis=0)
        return(ret)
        