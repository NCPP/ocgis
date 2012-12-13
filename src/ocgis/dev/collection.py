from ocgis.util.helpers import iter_array
from warnings import warn
from collections import OrderedDict


class OcgDimension(object):
    _bounds_prefix = 'bnds'
    
    def __init__(self,uid_name,uid,value_name,value,bounds=None):
        self.uid_name = uid_name
        self.uid = uid
        self.value_name = value_name
        self.value = value
        self.bounds = bounds
        
        template = '{0}{1}_{2}'
        self.headers = {'uid':self.uid_name,
                        'value':self.value_name,
                        'bnds':{0:template.format(self._bounds_prefix,0,self.value_name),
                                1:template.format(self._bounds_prefix,1,self.value_name)}}
        
    def iter_rows(self,add_bounds=True):
        uid = self.uid
        value = self.value
        bounds = self.bounds
        
        uid_name = 'uid'
        value_name = 'value'
        
        if add_bounds and bounds is None:
            warn('bounds requested in iteration, but no bounds variable exists.')
            add_bounds = False
        
        for (idx,) in iter_array(value):
            ret = {uid_name:uid[idx],
                   value_name:value[idx]}
            if add_bounds:
                ret.update({'bnds':{0:bounds[idx,0],
                                    1:bounds[idx,1]}})
            yield(ret)
            
            
class TemporalDimension(OcgDimension):
    
    def __init__(self,uid,value,bounds=None):
        super(TemporalDimension,self).__init__('tid',uid,'time',value,bounds=bounds)


class OcgIdentifier(OrderedDict):
    
    def __init__(self,*args,**kwds):
        self._curr = 1
        super(OcgIdentifier,self).__init__(*args,**kwds)
    
    def add(self,value):
        if self._get_is_unique_(value):
            self.update({value:self._get_current_identifier_()})
        
    def _get_is_unique_(self,value):
        if value in self:
            ret = False
        else:
            ret = True
        return(ret)
    
    def _get_current_identifier_(self):
        try:
            return(self._curr)
        finally:
            self._curr += 1


class OcgVariable(object):
    pass
