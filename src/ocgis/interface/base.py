from ocgis import env, constants
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import itertools
from collections import deque
from ocgis.exc import EmptyData
import datetime


class AbstractDataset(object):
    __metaclass__ = ABCMeta
    _name_value = 'value'
    
    @abstractproperty
    def _dtemporal(self): AbstractTemporalDimension
    @abstractproperty
    def _dlevel(self): AbstractLevelDimension
    @abstractproperty
    def _dspatial(self): AbstractSpatialDimension
    
    def __init__(self,request_dataset=None,temporal=None,level=None,spatial=None,
                 metadata=None,value=None):
        self.request_dataset = request_dataset
        
        self._temporal = temporal
        self._level = level
        self._spatial = spatial
        self._metadata = metadata
        self._value = value
        
        self._dummy_level = False
        self._dummy_temporal = False
        
    @abstractmethod
    def __getitem__(self): pass
        
    @property
    def level(self):
        if self._dlevel is None:
            self._dummy_level = True
        if self._level is None and not self._dummy_level:
            self._level = self._dlevel._load_(self)
            if self._level is None:
                self._dummy_level = True
        return(self._level)
    
    @abstractproperty
    def metadata(self): pass
    
    @property
    def temporal(self):
        if self._dtemporal is None:
            self._dummy_temporal = True
        if self._temporal is None and not self._dummy_temporal:
            self._temporal = self._dtemporal._load_(self)
            if self._temporal is None:
                self._dummy_temporal = True
        return(self._temporal)
    
    @property
    def spatial(self):
        if self._spatial is None:
            self._spatial = self._dspatial._load_(self)
            assert(self._spatial is not None)
        return(self._spatial)
    
    @property
    def value(self):
        return(self._value)
    
    def aggregate(self):
        msg = "Aggregation is not implemented for {0}".format(self.__class__.__name__)
        raise(NotImplementedError(msg))
    
    @abstractmethod
    def get_subset(self,temporal=None,level=None,spatial=None):
        pass
    
    @abstractmethod
    def project(self,projection):
        ''':type to_sr: ocgis.interface.OcgSpatialReference'''
        return(None)


class AbstractInterfaceDimension(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def extent(self): "<varying>"
    @abstractproperty
    def resolution(self): "<varying>"
    @abstractproperty
    def _name_id(self): str
    @abstractproperty
    def _name_long(self): str
    
    def __init__(self,subset_by=None,value=None,uid=None,bounds=None,
                 real_idx=None,name=None,name_bounds=None,dataset=None):
        
        if value is None and bounds is not None:
            raise(ValueError("Bounds must be passed with an associated value."))
        
        self.name = name
        self.name_bounds = name_bounds
        self.real_idx = real_idx
        self.uid = uid
        self.dataset = dataset
        
        ## ensure shape of incoming arrays
        if value is not None:
            self._value = np.atleast_1d(value)
        else:
            self._value = None
        if bounds is not None:
            self._bounds = np.atleast_2d(bounds)
        else:
            self._bounds = None
        
        ## pass the original argument values as this method calls the class load.
        ## if no value is present, we assume we need to load the data and go about
        ## calling the class _load_ methods.
        self._set_value_bounds_uid_real_idx_(value,bounds,uid,subset_by,real_idx)
    
    @abstractproperty
    def __getitem__(self): pass
    
    @abstractmethod
    def subset(self): pass
    
    @property
    def value(self):
        return(self._value)
    @value.setter
    def value(self,val):
        if val is not None:
            val = np.atleast_1d(val)
        self._value = val
        
    @property
    def bounds(self):
        return(self._bounds)
    @bounds.setter
    def bounds(self,val):
        if val is not None:
            val = np.atleast_2d(val)
        self._bounds = val
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        raise(NotImplementedError)
    
    def _set_value_bounds_uid_real_idx_(self,value,bounds,uid,subset_by,real_idx):
        if value is None:
            self.value,self.bounds,self.uid,self.real_idx = self._load_(subset_by=subset_by)
        if self.uid is None:
            self.uid = np.arange(1,self.shape[0]+1,dtype=int)
        if self.real_idx is None:
            self.real_idx = np.arange(0,self.shape[0],dtype=int)
            
            
class AbstractVectorDimension(object):
    __metaclass__ = ABCMeta
    
    def __getitem__(self,slc):
        value = np.atleast_1d(self.value[slc])
        if self.bounds is None:
            bounds = None
        else:
            bounds = np.atleast_2d(self.bounds[slc,:])
        uid = np.atleast_1d(self.uid[slc])
        real_idx = np.atleast_1d(self.real_idx[slc])
        ret = self.__class__(value=value,bounds=bounds,
                             uid=uid,real_idx=real_idx,name=self.name,
                             name_bounds=self.name_bounds,dataset=self.dataset)
        return(ret)
    
    @property
    def extent(self):
        if self.bounds is None:
            ret = (self.value.min(),self.value.max())
        else:
            ret = (self.bounds.min(),self.bounds.max())
        return(ret)
    
    @property
    def resolution(self):
        ret = np.abs(np.ediff1d(self.value).mean())
        return(ret)
    
    @property
    def shape(self):
        return(self.value.shape)
    
    def _get_iter_value_bounds_ref(self):
        return(self.value,self.bounds)
    
    def get_iter(self,add_bounds=True):
        uid = self.uid
        value,bounds = self._get_iter_value_bounds_ref()
        has_bounds = False if bounds is None else True
        if not add_bounds and has_bounds:
            has_bounds = False
        name_id = self._name_id
        name_value = self._name_long
        name_left_bound = 'bnd_left_'+name_value
        name_right_bound = 'bnd_right_'+name_value
        
        ret = {}
        ## get the shape in case it is singleton
        try:
            shp = value.shape[0]
        except AttributeError:
            value = np.array([value])
            uid = np.array([uid])
            bounds = np.array([bounds])
            shp = value.shape[0]
        for idx in range(shp):
            ret[name_value] = value[idx]
            ret[name_id] = uid[idx]
            if has_bounds:
                ret[name_left_bound] = bounds[idx,0]
                ret[name_right_bound] = bounds[idx,1]
            yield(idx,ret)
    
    def subset(self,lower,upper):
        if self.bounds is None:
            lidx = self.value >= lower
            uidx = self.value <= upper
            idx = np.logical_and(lidx,uidx)
            bounds = None
        else:
            ## identify ordering
            if self.bounds[0,0] > self.bounds[0,1]:
                lower_col = 1
                upper_col = 0
            else:
                lower_col = 0
                upper_col = 1
            lidx = self.bounds[:,upper_col] > lower
            uidx = self.bounds[:,lower_col] < upper
            idx = np.logical_and(lidx,uidx)
            if not idx.any():
                ## bounds may align with centroids, check if the centroids
                ## return a match
                lidx = self.value >= lower
                uidx = self.value <= upper
                idx = np.logical_and(lidx,uidx)
                if not idx.any():
                    raise(EmptyData('temporal subset returned empty'))
            bounds = np.atleast_2d(self.bounds[idx,:])
        ret = self.__class__(value=self.value[idx],bounds=bounds,
                             uid=self.uid[idx],real_idx=self.real_idx[idx],
                             name=self.name,name_bounds=self.name_bounds,
                             dataset=self.dataset)
        return(ret)


class AbstractLevelDimension(AbstractVectorDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractTemporalDimension(AbstractVectorDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    _date_parts = ('year','month','day','hour','minute','second','microsecond')
    
    def __init__(self,*args,**kwds):
        super(AbstractTemporalDimension,self).__init__(*args,**kwds)
        self.group = None
    
    @abstractproperty
    def _dtemporal_group_dimension(self): AbstractTemporalGroupDimension
    
    def _get_iter_value_bounds_ref(self):
        return(self.value_datetime,self.bounds_datetime)
    
    def set_grouping(self,grouping):
        
        ## if we are optimizing, attempt to pull the group
        if env.OPTIMIZE_FOR_CALC:
            self.group = env._optimize_store[self.dataset.request_dataset.alias]['group']
            return()
        
        group_map = dict(zip(range(0,7),self._date_parts,))
        group_map_rev = dict(zip(self._date_parts,range(0,7),))

        value = np.empty((self.value.shape[0],3),dtype=object)
        
        if self.bounds is None:
            value[:,:] = self.value_datetime.reshape(-1,1)
        else:
            value[:,0] = self.bounds_datetime[:,0]
            value[:,1] = self.value_datetime
            value[:,2] = self.bounds_datetime[:,1]
        
        def _get_attrs_(dt):
            return([dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond])
        
        parts = np.empty((len(self.value),len(self._date_parts)),dtype=int)
        for row in range(parts.shape[0]):
            parts[row,:] = _get_attrs_(value[row,1])
        
        unique = deque()
        for idx in range(parts.shape[1]):
            if group_map[idx] in grouping:
                fill = np.unique(parts[:,idx])
            else:
#                fill = np.array([0])
                fill = [None]
            unique.append(fill)

        select = deque()
        idx2_seq = range(len(self._date_parts))
        for idx in itertools.product(*[range(len(u)) for u in unique]):
            select.append([unique[idx2][idx[idx2]] for idx2 in idx2_seq])
        select = np.array(select)
        dgroups = deque()
        idx_cmp = [group_map_rev[group] for group in grouping]
        keep_select = []
        for idx in range(select.shape[0]):
            match = select[idx,idx_cmp] == parts[:,idx_cmp]
            dgrp = match.all(axis=1)
            if dgrp.any():
                keep_select.append(idx)
                dgroups.append(dgrp)
        select = select[keep_select,:]
        assert(len(dgroups) == select.shape[0])
        
        dtype = [(dp,object) for dp in self._date_parts]
        new_value = np.empty((len(dgroups),),dtype=dtype)
        new_bounds = np.empty((len(dgroups),2),dtype=object)

        for idx,dgrp in enumerate(dgroups):
            ## tuple conversion is required for structure arrays: http://docs.scipy.org/doc/numpy/user/basics.rec.html#filling-structured-arrays
            new_value[idx] = tuple(select[idx])
            sel = value[dgrp][:,(0,2)]
            new_bounds[idx,:] = [sel.min(),sel.max()]
        
        new_bounds = np.atleast_2d(new_bounds)
        new_value = np.atleast_1d(new_value)
        self.group = self._dtemporal_group_dimension(grouping,new_value,new_bounds,dgroups)

    
class AbstractTemporalGroupDimension(AbstractVectorDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    def __init__(self,grouping,value,bounds,dgroups,uid=None):
        self.grouping = grouping
        self.value = np.atleast_1d(value)
        assert(isinstance(self.value,np.ndarray))
        self.bounds = np.atleast_2d(bounds)
        assert(isinstance(self.bounds[0,0],datetime.datetime))
        self.dgroups = dgroups
        if uid is None:
            uid = np.arange(1,self.value.shape[0]+1,dtype=int)
        self.uid = uid
        self._representative_datetime = None
    
    @property
    def representative_datetime(self):
        if self._representative_datetime is None:
            ref_value = self.value
            ref_bounds = self.bounds
            ret = np.empty((ref_value.shape[0],),dtype=object)
            set_grouping = set(self.grouping)
            if set_grouping == set(['month']):
                ref_calc_month_centroid = constants.calc_month_centroid
                for idx in range(ret.shape[0]):
                    month = ref_value[idx]['month']
                    ## get the start year from the bounds data
                    start_year = ref_bounds[idx][0].year
                    ## create the datetime object
                    ret[idx] = datetime.datetime(start_year,month,ref_calc_month_centroid)
            elif set_grouping == set(['year']):
                ref_calc_year_centroid_month = constants.calc_year_centroid_month
                ref_calc_year_centroid_day = constants.calc_year_centroid_day
                for idx in range(ret.shape[0]):
                    year = ref_value[idx]['year']
                    ## create the datetime object
                    ret[idx] = datetime.datetime(year,ref_calc_year_centroid_month,ref_calc_year_centroid_day)
            elif set_grouping == set(['month','year']):
                ref_calc_month_centroid = constants.calc_month_centroid
                for idx in range(ret.shape[0]):
                    year,month = ref_value[idx]['year'],ref_value[idx]['month']
                    ret[idx] = datetime.datetime(year,month,ref_calc_month_centroid)
            elif set_grouping == set(['day']):
                for idx in range(ret.shape[0]):
                    start_year,start_month = ref_bounds[idx][0].year,ref_bounds[idx][0].month
                    ret[idx] = datetime.datetime(start_year,start_month,ref_value[idx]['day'],12)
            elif set_grouping == set(['day','month']):
                for idx in range(ret.shape[0]):
                    start_year = ref_bounds[idx][0].year
                    day,month = ref_value[idx]['day'],ref_value[idx]['month']
                    ret[idx] = datetime.datetime(start_year,month,day,12)
            elif set_grouping == set(['day','year']):
                for idx in range(ret.shape[0]):
                    day,year = ref_value[idx]['day'],ref_value[idx]['year']
                    ret[idx] = datetime.datetime(year,1,day,12)
            elif set_grouping == set(['day','year','month']):
                for idx in range(ret.shape[0]):
                    day,year,month = ref_value[idx]['day'],ref_value[idx]['year'],ref_value[idx]['month']
                    ret[idx] = datetime.datetime(year,month,day,12)
            else:
                raise(NotImplementedError('grouping: {0}'.format(self.grouping)))
            return(ret)
        
    def get_iter(self,add_bounds=True):
        value = self.value
        uid = self.uid
        bounds = self.bounds
        has_bounds = False if bounds is None else True
        if not add_bounds and has_bounds:
            has_bounds = False
        name_id = self._name_id
        
        ret = {}
        for idx in range(value.shape[0]):
            ret[name_id] = uid[idx]
            ret['year'] = value[idx]['year']
            ret['month'] = value[idx]['month']
            ret['day'] = value[idx]['day']
            yield(idx,ret)
    
    
class AbstractRowDimension(AbstractVectorDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractColumnDimension(AbstractRowDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractSpatialDimension(object):
    __metaclass__ = ABCMeta
    
    def __init__(self,projection=None):
        self.projection = projection
    
    @abstractproperty
    def weights(self): np.ma.MaskedArray
    
    @abstractmethod
    def get_iter(self): pass
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        raise(NotImplementedError)


class AbstractSpatialGrid(AbstractSpatialDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    @property
    def weights(self):
        if self._weights is None:
            self._weights = np.ones(self.shape,dtype=float)
        return(self._weights)


class AbstractSpatialVector(AbstractSpatialDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    _name_id = None
    _name_long = None
    
    def __getitem__(self,slc):
        raise(NotImplementedError)
    
    @property
    def geom(self):
        if self._geom is None:
            self._geom = self._get_all_geoms_()
        return(self._geom)
    
    @property
    def resolution(self):
        raise(NotImplementedError('Resolution is not a spatial vector property.'))
    
    @property
    def shape(self):
        return(self.geom.shape)
    
    @abstractmethod
    def clip(self,polygon): pass
    
    @abstractmethod
    def intersects(self,polygon): pass
    
    def subset(self):
        raise(NotImplementedError('Use "intersects" or "clip".'))
    
    @abstractmethod
    def unwrap(self): pass
    
    @abstractmethod
    def wrap(self): pass
    
    @abstractmethod
    def _get_all_geoms_(self): np.ma.MaskedArray


class AbstractPointDimension(AbstractSpatialVector):
    __metaclass__ = ABCMeta
    

class AbstractPolygonDimension(AbstractSpatialVector):
    __metaclass__ = ABCMeta 