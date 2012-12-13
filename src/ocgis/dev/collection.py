from ocgis.util.helpers import iter_array
from warnings import warn
from collections import OrderedDict, deque
import datetime
import numpy as np
import itertools


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
        
    def group(self,*args):
        '''
        ('month',2)
        (['month','year'])
        '''
        if len(args) == 2:
            return(self._group_part_count_(*args))
        else:
            return(self._group_part_(*args))
    
    def _group_part_count_(self,part,count):
        raise(NotImplementedError)
        if part not in ['year','month']:
            raise(NotImplementedError)
            try:
                delta = datetime.timedelta(**{part:count})
            except TypeError:
                delta = datetime.timedelta(**{part+'s':count})
            lower = value[0]
            upper = lower + delta
                
        value = self.value
        bounds = self.bounds
        
        if self.bounds is None:
            raise(NotImplementedError)
        else:
            ## get exclusive lower and upper bounds
            lower = getattr(value[0],part)
            upper = lower + count
            import ipdb;ipdb.set_trace()
        
    def _subset_timeidx_(self,time_range):
        if time_range is None:
            ret = self.timeidx
        else:
            if self.bounds is None:
                ret = self.timeidx[(self.value>=time_range[0])*
                                   (self.value<=time_range[1])]
            else:
                select = np.empty(self.value.shape,dtype=bool)
                for idx in np.arange(self.bounds.shape[0]):
                    bnds = self.bounds[idx,:]
                    idx1 = (time_range[0]>=bnds[0])*(time_range[0]<=bnds[1])
                    idx2 = (time_range[0]<=bnds[0])*(time_range[1]>=bnds[1])
                    idx3 = (time_range[1]>=bnds[0])*(time_range[1]<=bnds[1])
                    select[idx] = np.logical_or(np.logical_or(idx1,idx2),idx3)
                ret = self.timeidx[select]
        return(ret)
    
    def _group_part_(self,groups):
        
        if self.bounds is None:
            vrshp = self.value.reshape(-1,1)
            dval = np.hstack((vrshp,vrshp,vrshp))
        else:
            dval = np.empty((self.value.shape[0],3),dtype=object)
            dval[:,0] = self.bounds[:,0]
            dval[:,1] = self.value
            dval[:,2] = self.bounds[:,1]
        
        date_parts = ('year','month','day','hour','minute','second','microsecond')
        group_map = dict(zip(range(0,7),date_parts,))
        
        def _get_attrs_(dt,groups):
            return([getattr(dt,group) for group in groups])
            return([dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond])
        
        parts = np.empty((len(self.value),3,7),dtype=int)
        for row,col in itertools.product(range(parts.shape[0]),range(parts.shape[1])):
            parts[row,col,:] = _get_attrs_(dval[row,col])
        
        unique = deque()
        for idx in range(parts.shape[2]):
            if group_map[idx] in groups:
                fill = np.unique(parts[:,:,idx])
            elif idx == 2:
                fill = np.array([1])
            else:
                fill = np.array([0])
            unique.append(fill)
        
        select = deque()
        idx2_seq = range(7)
        for idx in itertools.product(*[range(len(u)) for u in unique]):
            try:
                select.append(datetime.datetime(*[unique[idx2][idx[idx2]] for idx2 in idx2_seq]))
            except ValueError:
                continue
            
        import ipdb;ipdb.set_trace()
#            arr = np.empty((len(value),3),dtype=int)
#            for idx in range(len(value)):
#                arr[idx,1] = getattr(value[idx],group)
#                if bounds is None:
#                    arr[idx,::2] = arr[idx,1]
#                else:
#                    arr[idx,0] = getattr(bounds[idx,0],group)
#                    arr[idx,2] = getattr(bounds[idx,1],group)
#            arrs[group] = arr
#        import ipdb;ipdb.set_trace()
        
class TemporalGroupDimension(OcgDimension):
    
    def __init__(self,uid,value,bounds,groups):
        super(TemporalDimension,self).__init__('tgid',uid,'time',value,bounds)
        
        self.groups = groups

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
