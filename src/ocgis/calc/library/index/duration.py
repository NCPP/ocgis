from ocgis.calc import base
import numpy as np
from ocgis.util.helpers import iter_array
from ocgis.exc import DefinitionValidationError
from ocgis import constants
from collections import OrderedDict


class Duration(base.AbstractUnivariateSetFunction,base.AbstractParameterizedFunction):
    key = 'duration'
    parms_definition = {'threshold':float,'operation':str,'summary':str}
    ## output data type will vary by the summary operation (e.g. float for mean,
    ## int for max)
    dtype = constants.np_float
    description = 'Summarizes consecutive occurrences in a sequence where the logical operation returns TRUE. The summary operation is applied to the sequences within a temporal aggregation.'
    standard_name = 'duration'
    long_name = 'Duration'

    def calculate(self,values,threshold=None,operation=None,summary='mean'):
        '''
        :param threshold: The threshold value to use for the logical operation.
        :type threshold: float
        :param operation: The logical operation. One of 'gt','gte','lt', or 'lte'.
        :type operation: str
        :param summary: The summary operation to apply the durations. One of 'mean','median','std','max', or 'min'.
        :type summary: str
        '''
        assert(len(values.shape) == 3)
        ## storage array for counts
        shp_out = values.shape[-2:]
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
        
        ## update the output mask. this only applies to geometries so pick the
        ## first masked time field
        store = np.ma.array(store,mask=values.mask[0,:,:])
                
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
        for rowidx,colidx in iter_array(values[0,:,:],use_mask=False):
            vec = arr[:,rowidx,colidx].reshape(-1)
            ## check first if there is a longer series than 1
            if np.any(np.diff(ref[vec]) == 1):
                ## find locations where the values switch
                diff_idx = np.diff(vec)
                if diff_idx.shape != ref.shape:
                    diff_idx = np.append(diff_idx,[False])
                split_idx = ref[diff_idx] + 1
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


class FrequencyDuration(base.AbstractKeyedOutputFunction,Duration):
    key = 'freq_duration'
    description = 'Count the frequency of spell durations within the temporal aggregation.'
    dtype = object
    structure_dtype = OrderedDict([['names',['duration','count']],['formats',[constants.np_int,constants.np_int]]])
    parms_definition = {'threshold':float,'operation':str}
    standard_name = 'frequency_duration'
    long_name = 'Frequency Duration'
    
    def calculate(self,values,threshold=None,operation=None):
        '''
        :param threshold: The threshold value to use for the logical operation.
        :type threshold: float
        :param operation: The logical operation. One of 'gt','gte','lt', or 'lte'.
        :type operation: str
        '''
        shp_out = values.shape[-2:]
        store = np.zeros(shp_out,dtype=object).flatten()
        for ii,duration in enumerate(self._iter_consecutive_(values,threshold,operation)):
            summary = self._get_summary_(duration)
            store[ii] = summary
        store.resize(shp_out)
        
        ## update the output mask. this only applies to geometries so pick the
        ## first masked time field
        store = np.ma.array(store,mask=values.mask[0,:,:])
        
        return(store)
        
    def _get_summary_(self,duration):
        '''
        :param duration: List of duration elements for frequency target.
        :type duration: list
        
        >>> duration = [3, 5, 2, 2]
        
        :returns: NumPy structure with dimension equal to the count of unique elements
         in the `duration` sequence.
        '''
        set_duration = set(duration)
        ret = np.empty(len(set_duration),dtype=self.structure_dtype)
        for ii,sd in enumerate(set_duration):
            idx = np.array(duration) == sd
            count = idx.sum()
            ret[ii]['duration'] = sd
            ret[ii]['count'] = count
        return(ret)
    
    @classmethod
    def validate(cls,ops):
        Duration.validate(ops)
