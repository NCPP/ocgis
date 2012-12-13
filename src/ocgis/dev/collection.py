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
        
        for idx in range(value.shape[0]):
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
            new_value,new_bounds,dgroups = self._group_part_count_(*args)
        else:
            new_value,new_bounds,dgroups = self._group_part_(*args)
        
        uid = np.arange(1,len(new_value)+1)
        
        return(TemporalGroupDimension(uid,new_value,new_bounds,dgroups))
    
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
        
        date_parts = ('year','month','day','hour','minute','second',)
        group_map = dict(zip(range(0,7),date_parts,))
        group_map_rev = dict(zip(date_parts,range(0,7),))
        
        if self.bounds is None:
            vrshp = self.value.reshape(-1,1)
            dval = np.hstack((vrshp,vrshp))
        else:
            dval = self.bounds.copy()
            delta = datetime.timedelta(microseconds=1)
            for idx in range(dval.shape[0]):
                dval[idx,1] = dval[idx,1]-delta

        def _get_attrs_(dt):
            return([dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,])
        parts = np.empty((len(self.value),2,len(date_parts)),dtype=int)
        for row,col in itertools.product(range(parts.shape[0]),range(parts.shape[1])):
            parts[row,col,:] = _get_attrs_(dval[row,col])
        
        unique = deque()
        for idx in range(parts.shape[2]):
            if group_map[idx] in groups:
                fill = np.unique(parts[:,:,idx])
            else:
                fill = np.array([0])
            unique.append(fill)
        
        select = deque()
        idx2_seq = range(len(date_parts))
        for idx in itertools.product(*[range(len(u)) for u in unique]):
            select.append([unique[idx2][idx[idx2]] for idx2 in idx2_seq])
        select = np.array(select)

        dgroups = deque()
        idx_cmp = [group_map_rev[group] for group in groups]
        for idx in range(select.shape[0]):
            match = select[idx,idx_cmp] == parts[:,:,idx_cmp]
            dgrp = np.any(np.all(match,axis=2),axis=1)
            if dgrp.any():
                dgroups.append(dgrp)
            
        new_value = np.empty((len(dgroups),len(date_parts)),dtype=int)
        new_bounds = np.empty((len(dgroups),2),dtype=object)
        if self.bounds is None:
            bounds = dval
        else:
            bounds = self.bounds
        for idx,dgrp in enumerate(dgroups):
            sel = bounds[dgrp]
            new_value[idx] = select[idx]
            new_bounds[idx,:] = [sel.min(),sel.max()]

        return(new_value,new_bounds,dgroups)
        
class TemporalGroupDimension(OcgDimension):
    
    def __init__(self,uid,value,bounds,dgroups):
        super(TemporalGroupDimension,self).__init__('tgid',uid,None,value,bounds)
        
        self.dgroups = dgroups

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
