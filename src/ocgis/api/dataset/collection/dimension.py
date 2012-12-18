from ocgis.util.helpers import get_bounded, iter_array
import numpy as np
from shapely.geometry.point import Point
from collections import deque
import itertools
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkb


class OcgDimension(object):
    _name_value = 'value'
    _name_uid = 'uid'
    
    def __init__(self,uid,value,bounds=None):
        self.storage = get_bounded(value,bounds,uid,
                        names={'uid':self._name_uid,'value':self._name_value})
        
    @property
    def shape(self):
        return(self.storage.shape)
    @property
    def uid(self):
        return(self.storage[self._name_uid])
    @property
    def value(self):
        return(self.storage[self._name_value])
    
    def __len__(self):
        return(self.storage.shape[0])
    
    def __iter__(self):
        _name_uid = self._name_uid
        _name_value = self._name_value
        value = self.value
        uid = self.uid
        
        for idx in range(value.shape[0]):
            row = {_name_uid:uid[idx],
                   _name_value:value[idx,0]}
            yield(idx,row)
        
#    def __iter__(self):
#        storage = self.storage
#        for idx in range(storage.shape[0]):
#            yield(idx,storage[idx])
    
#    def iter_rows(self,add_bounds=True):
##        name_value = self._name_value
##        name_bounds = self._name_bounds
##        name_uid = self._name_uid
##        
##        if add_bounds and bounds is None:
##            warn('bounds requested in iteration, but no bounds variable exists.')
##            add_bounds = False
#        
#        for idx in self._iter_values_idx_(value):
#            ret = {name_value:value[idx],name_uid:uid[idx]}
#            if add_bounds:
#                ret.update({name_bounds:{0:bounds[idx,0],
#                                         1:bounds[idx,1]}})
#            yield(yld)
#    
#    def _iter_values_idx_(self,value):
#        for idx in range(value.shape[0]):
#            yield(idx)
            
            
class LevelDimension(OcgDimension):
    _name_value = 'level'
    _name_uid = 'lid'
    

class SpatialDimension(OcgDimension):
    _name_value = 'geom'
    _name_uid = 'gid'
    
    def __init__(self,uid,value,value_mask,weights=None):
        self._uid = uid
        self._value = value
        self._value_mask = value_mask
        
        if weights is None:
            if len(value) > 0:
                weights = self._get_weights_()
            else:
                weights = None
        else:
            assert(weights.shape == value.shape)
        self.weights = weights
    
    @property
    def uid(self):
        return(self._uid)
    @property
    def value(self):
        return(np.ma.array(self._value,mask=self._value_mask))
    @property
    def bounds(self):
        raise(NotImplementedError)
    @property
    def geomtype(self):
        if isinstance(self._value[0,0],Point):
            ret = 'point'
        else:
            ret = 'polygon'
        return(ret)
    @property
    def shape(self):
        return(self._value.shape)
    
    def __len__(self):
        return(self.value.compressed().shape[0])
    
    def __iter__(self):
        _name_uid = self._name_uid
        _name_value = self._name_value
        uid = self.uid
        _conv_to_multi_ = self._conv_to_multi_
        
        for idx,geom in iter_array(self.value,return_value=True):
            row = {_name_uid:uid[idx],
                   _name_value:_conv_to_multi_(geom)}
            yield(idx,row)
            
    def _get_weights_(self):
        value = self._value
        value_mask = self._value_mask
        
        if isinstance(self._value[0,0],Point):
            weights = np.ones(value.shape,dtype=float)
            weights = np.ma.array(weights,mask=value_mask)
        else:
            weights = np.empty(value.shape,dtype=float)
            masked = self.value
            for idx,geom in iter_array(masked,return_value=True):
                weights[idx] = geom.area
            weights = weights/weights.max()
            weights = np.ma.array(weights,mask=value_mask)
        return(weights)
    
    @staticmethod
    def _conv_to_multi_(geom):
        '''Geometry conversion to single type.'''
        
        if isinstance(geom,Point):
            pass
        else:
            try:
                geom = MultiPolygon(geom)
            except TypeError:
                geom = MultiPolygon([geom])
            except AssertionError:
                geom = wkb.loads(geom.wkb)
        return(geom)
            
        
#    def get_masked(self):
#        return(np.ma.array(self.value,mask=self.value_mask))
#    
#    def _iter_values_idx_(self,value):
#        for idx in iter_array(self.get_masked()):
#            yield(idx)


class TemporalDimension(OcgDimension):
    _name_value = 'time'
    _name_uid = 'tid'
    
#    def __init__(self,uid,value,bounds=None):
#        super(TemporalDimension,self).__init__(uid,value,bounds=bounds)
        
#        self.tgdim = None
    
    def group(self,*args):
        '''
        ('month',2)
        (['month','year'])
        '''
        if len(args) == 2:
            new_value,new_bounds,dgroups = self._group_part_count_(*args)
        else:
            new_value,new_bounds,dgroups = self._group_part_(*args)
        
        return(TemporalGroupDimension(new_value,new_bounds,dgroups,args[0]))
    
#    def _group_part_count_(self,part,count):
#        raise(NotImplementedError)
#        if part not in ['year','month']:
#            raise(NotImplementedError)
#            try:
#                delta = datetime.timedelta(**{part:count})
#            except TypeError:
#                delta = datetime.timedelta(**{part+'s':count})
##            lower = value[0]
##            upper = lower + delta
#                
#        value = self.value
#        bounds = self.bounds
#        
#        if self.bounds is None:
#            raise(NotImplementedError)
#        else:
#            ## get exclusive lower and upper bounds
#            lower = getattr(value[0],part)
#            upper = lower + count
#            import ipdb;ipdb.set_trace()
#        
#    def _subset_timeidx_(self,time_range):
#        if time_range is None:
#            ret = self.timeidx
#        else:
#            if self.bounds is None:
#                ret = self.timeidx[(self.value>=time_range[0])*
#                                   (self.value<=time_range[1])]
#            else:
#                select = np.empty(self.value.shape,dtype=bool)
#                for idx in np.arange(self.bounds.shape[0]):
#                    bnds = self.bounds[idx,:]
#                    idx1 = (time_range[0]>=bnds[0])*(time_range[0]<=bnds[1])
#                    idx2 = (time_range[0]<=bnds[0])*(time_range[1]>=bnds[1])
#                    idx3 = (time_range[1]>=bnds[0])*(time_range[1]<=bnds[1])
#                    select[idx] = np.logical_or(np.logical_or(idx1,idx2),idx3)
#                ret = self.timeidx[select]
#        return(ret)
    
    def _group_part_(self,groups):
        
        date_parts = ('year','month','day','hour','minute','second','microsecond')
        group_map = dict(zip(range(0,7),date_parts,))
        group_map_rev = dict(zip(date_parts,range(0,7),))
        value = self.value

        def _get_attrs_(dt):
            return([dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond])
        parts = np.empty((len(self.value),len(date_parts)),dtype=int)
        for row in range(parts.shape[0]):
            parts[row,:] = _get_attrs_(value[row,1])
        
        unique = deque()
        for idx in range(parts.shape[1]):
            if group_map[idx] in groups:
                fill = np.unique(parts[:,idx])
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
        keep_select = []
        for idx in range(select.shape[0]):
            match = select[idx,idx_cmp] == parts[:,idx_cmp]
            dgrp = match.all(axis=1)
            if dgrp.any():
                keep_select.append(idx)
                dgroups.append(dgrp)
        select = select[keep_select,:]
        assert(len(dgroups) == select.shape[0])
        
        new_value = np.empty((len(dgroups),len(date_parts)),dtype=int)
        new_bounds = np.empty((len(dgroups),2),dtype=object)
#        if self.bounds is None:
#            bounds = value
#        else:
#            bounds = self.bounds
        for idx,dgrp in enumerate(dgroups):
            new_value[idx] = select[idx]
            sel = value[dgrp][:,(0,2)]
            new_bounds[idx,:] = [sel.min(),sel.max()]

        return(new_value,new_bounds,dgroups)
    

class TemporalGroupDimension(OcgDimension):
    _name_value = None
    _name_uid = 'tgid'
    _date_parts = ('year','month','day','hour','minute','second','microsecond')
    
    def __init__(self,value,bounds,dgroups,groups):
        self._uid = np.arange(1,value.shape[0]+1)
        self._value = value
        self.bounds = bounds
        self.groups = groups
        self.dgroups = dgroups
        
    @property
    def uid(self):
        return(self._uid)
    @property
    def value(self):
        return(self._value)
    @property
    def shape(self):
        return(self._value.shape)
    
    def __len__(self):
        return(self.value.shape[0])
        
    def __iter__(self):
        value = self.value
#        bounds = self.bounds
        get_idx = [self._date_parts.index(g) for g in self.groups]
        groups = self.groups
        uid = self.uid
        
        for idx in range(value.shape[0]):
            ret = dict(zip(groups,[value[idx,gi] for gi in get_idx]))
            ret.update({'tgid':uid[idx]})
#            ret.update({'lower':bounds[idx,0],
#                        'upper':bounds[idx,1]})
            yield(idx,ret)
