from ocgis.util.helpers import iter_array
from warnings import warn


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
        
    def iter_rows(self,add_bounds=False):
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


class OcgIdentifier(object):
    pass


class OcgVariable(object):
    pass