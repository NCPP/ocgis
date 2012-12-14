from warnings import warn


class OcgDimension(object):
    _bounds_prefix = 'bnds'
    _value_name = 'value'
    
    def __init__(self,value,bounds=None):
        self.value = value
        self.bounds = bounds
    
    def iter_rows(self,add_bounds=True):
        value = self.value
        bounds = self.bounds
        value_name = self._value_name
        
        if add_bounds and bounds is None:
            warn('bounds requested in iteration, but no bounds variable exists.')
            add_bounds = False
        
        for idx in self._iter_values_idx_(value):
            ret = {value_name:value[idx]}
            if add_bounds:
                ret.update({'bnds':{0:bounds[idx,0],
                                    1:bounds[idx,1]}})
            yield(ret)
    
    @staticmethod
    def _iter_values_idx_(value):
        for idx in range(value.shape[0]):
            yield(idx)
            
            
class LevelDimension(OcgDimension):
    pass