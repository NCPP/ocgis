import groups
from base import OcgFunction, OcgCvArgFunction, OcgArgFunction
import numpy as np
from ocgis.util.helpers import iter_array


class FrequencyPercentile(OcgArgFunction):
    name = 'freq_perc'
    nargs = 2
    Group = groups.Percentiles
    dtype = np.float32
    description = 'Percentile value matching "perc".'
    
    def _calculate_(self,values,perc=None):
        perc = int(perc)
        ret = np.percentile(values,perc,axis=0)
        return(ret)


class SampleSize(OcgFunction):
    '''
    .. note:: Automatically added by OpenClimateGIS. This should generally not be invoked manually.
    '''
    name = 'n'
    description = 'Statistical sample size.'
    Group = groups.BasicStatistics
    dtype = np.int32
    
    def _calculate_(self,values):
        ret = np.empty(values.shape[-2:],dtype=int)
        ret[:] = values.shape[0]
        ret = np.ma.array(ret,mask=values.mask[0,0,:])
        return(ret)
    
    def _aggregate_spatial_(self,values,weights):
        return(np.ma.sum(values))


class Median(OcgFunction):
    description = 'Median value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.median(values,axis=0))
    
    
class Mean(OcgFunction):
    description = 'Mean value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.mean(values,axis=0))
    
    
class Max(OcgFunction):
    description = 'Max value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.max(values,axis=0))
    
    
class Min(OcgFunction):
    description = 'Min value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.min(values,axis=0))
    
    
class StandardDeviation(OcgFunction):
    description = 'Standard deviation for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    name = 'std'
    
    def _calculate_(self,values):
        return(np.ma.std(values,axis=0))


class MaxConsecutive(OcgArgFunction):
    name = 'max_cons'
    nargs = 2
    Group = groups.Thresholds
    dtype = np.int32
    description = ('Maximum number of consecutive occurrences in the sequence'
                   ' where the logical operation returns TRUE.')
    
    def _calculate_(self,values,threshold=None,operation=None):
        ## time index reference
        ref = np.arange(0,values.shape[0])
        ## storage array for counts
        store = np.empty(list(values.shape)[1:])
        ## perform requested logical operation
        if operation == 'gt':
            arr = values > threshold
        elif operation == 'lt':
            arr = values < threshold
        elif operation == 'gte':
            arr = values >= threshold
        elif operation == 'lte':
            arr = values <= threshold

        ## find longest sequence for each geometry across time dimension
        for xidx,yidx in iter_array(values[0,:]):
            vec = arr[:,xidx,yidx]
#            ## collapse data if no axis provided
#            if axis is None:
#                vec = vec.reshape(-1)
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
        
#        ## summarize across geometries if axis is collapsed
#        if axis is None:
#            store = np.max(store)
            
        return(store)
        

class Between(OcgArgFunction):
    nargs = 2
    description = 'Count of values falling within the limits lower and upper (inclusive).'
    Group = groups.Thresholds
    dtype = np.int32
    
    def _calculate_(self,values,lower=None,upper=None):
        idx = (values >= lower)*(values <= upper)
        return(np.ma.sum(idx,axis=0))
    
    
class Threshold(OcgArgFunction):
    nargs = 2
    description = 'Count of values where the logical operation is True.'
    Group = groups.Thresholds
    dtype = np.int32
    
    def _calculate_(self,values,threshold=None,operation=None):
        threshold = float(threshold)
        
        ## perform requested logical operation
        if operation == 'gt':
            idx = values > threshold
        elif operation == 'lt':
            idx = values < threshold
        elif operation == 'gte':
            idx = values >= threshold
        elif operation == 'lte':
            idx = values <= threshold
        else:
            raise(NotImplementedError('The operation "{0}" was not recognized.'.format(operation)))
        
        ret = np.ma.sum(idx,axis=0)
        return(ret)
        
    def _aggregate_spatial_(self,values,weights):
        return(np.ma.sum(values))
    

class HeatIndex(OcgCvArgFunction):
    description = 'Heat Index following: http://en.wikipedia.org/wiki/Heat_index. If temperature is < 80F or relative humidity is < 40%, the value is masked during calculation. Output units are Fahrenheit.'
    Group = groups.MultivariateStatistics
    dtype = np.float32
    nargs = 2
    keys = ['tas','rhs']
    name = 'heat_index'
    
    def _calculate_(self,tas=None,rhs=None,units=None):
        if units == 'k':
            tas = 1.8*(tas - 273.15) + 32
        else:
            raise(NotImplementedError)
        
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -6.83783e-3
        c6 = -5.481717e-2
        c7 = 1.22874e-3
        c8 = 8.5282e-4
        c9 = -1.99e-6
        
        idx = tas < 80
        tas.mask = np.logical_or(idx,tas.mask)
        idx = rhs < 40
        rhs.mask = np.logical_or(idx,rhs.mask)
        
        tas_sq = np.square(tas)
        rhs_sq = np.square(rhs)
        
        hi = c1 + c2*tas + c3*rhs + c4*tas*rhs + c5*tas_sq + c6*rhs_sq + \
             c7*tas_sq*rhs + c8*tas*rhs_sq + c9*tas_sq*rhs_sq
        
        return(hi)
