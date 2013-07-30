import groups
from base import OcgFunction, OcgCvArgFunction, OcgArgFunction
import numpy as np
from ocgis.util.helpers import iter_array
from ocgis.calc.base import KeyedFunctionOutput, ProtectedFunction
from ocgis.constants import np_int
from ocgis.exc import DefinitionValidationError


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


class Duration(ProtectedFunction,OcgArgFunction):
    name = 'duration'
    nargs = 3
    Group = groups.Thresholds
    dtype = np.float32
    description = ('Summarizes consecutive occurrences in a sequence where the logical operation returns TRUE. The summary operation is applied within the temporal aggregation.')
    
    def _calculate_(self,values,threshold=None,operation=None,summary='mean'):
        ## storage array for counts
        shp_out = list(values.shape)
        shp_out[0] = 1
        store = np.zeros(shp_out,dtype=self.dtype).flatten()
        ## get the summary operation from the numpy library
        summary_operation = getattr(np,summary)

        ## find longest sequence for each geometry across time dimension
        for ii,fill in enumerate(self._iter_consecutive_(values,threshold,operation)):
            ## case of only a singular occurrence
            if len(fill) > 1:
                fill = summary_operation(fill)
            else:
                try:
                    fill = fill[0]
                ## value is likely masked
                except IndexError:
                    fill = 0
            store[ii] = fill
        
        store.resize(shp_out)
        return(store)
    
    def _iter_consecutive_(self,values,threshold,operation):
        ## time index reference
        ref = np.arange(0,values.shape[0])
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
        for zidx,rowidx,colidx in iter_array(values[0,:,:,:],use_mask=False):
            vec = arr[:,zidx,rowidx,colidx]
            ## check first if there is a longer series than 1
            if np.any(np.diff(ref[vec]) == 1):
                split_idx = ref[np.diff(vec)] + 1
                splits = np.array_split(vec,split_idx)
                fill = [a.sum() for a in splits if np.all(a)]
            ## case of only a singular occurrence
            elif np.any(vec):
                fill = [1]
            ## case for no occurrence
            else:
                fill = [0]
            
            yield(fill)
    
    @classmethod 
    def validate(cls,ops):
        if 'year' not in ops.calc_grouping:
            msg = 'Calculation grouping must include "year" for duration calculations.'
            raise(DefinitionValidationError('calc',msg))
    
    
class FrequencyDuration(KeyedFunctionOutput,Duration):
    name = 'freq_duration'
    description = 'Counts the frequency of spell durations within the temporal aggregation.'
    nargs = 2
    dtype = object
    output_keys = ['duration','count']
    
    def _calculate_(self,values,threshold=None,operation=None):
        shp_out = list(values.shape)
        shp_out[0] = 1
        store = np.zeros(shp_out,dtype=self.dtype).flatten()
        for ii,duration in enumerate(self._iter_consecutive_(values,threshold,operation)):
            summary = self._get_summary_(duration)
            store[ii] = summary
        store.resize(shp_out)
        return(store)
        
    def _get_summary_(self,duration):
        set_duration = set(duration)
        ret = np.empty(len(set_duration),dtype=[('duration',np_int),('count',np_int)])
        for ii,sd in enumerate(set_duration):
            idx = np.array(duration) == sd
            count = idx.sum()
            ret[ii]['duration'] = sd
            ret[ii]['count'] = count
        return(ret)


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
