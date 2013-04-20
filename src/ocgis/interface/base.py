from ocgis import env
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class AbstractGlobalInterface(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def _dtemporal(self): AbstractTemporalDimension
    @abstractproperty
    def _dlevel(self): AbstractLevelDimension
    @abstractproperty
    def _dspatial(self): AbstractSpatialDimension
#    @abstractproperty
#    def _metadata_cls(self): AbstractMetadata
    
    def __init__(self,request_dataset=None,temporal=None,level=None,spatial=None,
                 metadata=None):
        self.request_dataset = request_dataset
        self._temporal = temporal
        self._level = level
        self._spatial = spatial
        self._metadata = metadata
        
    @property
    def level(self):
        if self._level is None:
            self._level = self._dlevel(gi=self)
        return(self._level)
    
    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._metadata_cls(gi=self)
        return(self._metadata)
    
    @property
    def temporal(self):
        if self._temporal is None:
            self._temporal = self._dtemporal(gi=self)
        return(self._temporal)
    
    @property
    def spatial(self):
        if self._spatial is None:
            self._spatial = self._dspatial(gi=self)
        return(self._spatial)

    def subset_by_dimension(self,temporal=None,level=None,spatial=None):
        return(self.get_dataset(temporal=None,level=None,spatial=None))
    
    
class AbstractInterfaceDimension(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def extent(self): "<varying>"
    @abstractproperty
    def resolution(self): "<varying>"
    @abstractproperty
    def shape(self): (int,)
    @abstractproperty
    def _name_id(self): str
    @abstractproperty
    def _name_long(self): str
    
    def __init__(self,gi=None,subset_by=None,value=None,uid=None,bounds=None,
                 real_idx=None):
        if value is None and bounds is not None:
            raise(ValueError("Bounds must be passed with an associated value."))
        
        self.gi = gi
        self.value = value
        self.bounds = bounds
        self.real_idx = real_idx
        self.uid = uid
        self._set_value_bounds_uid_(value,bounds,uid,subset_by,real_idx)
        
    @abstractmethod
    def subset(self): pass
        
    @abstractmethod
    def _load_(self,subset_by=None):
        return("do some operation here")
    
    def _set_value_bounds_uid_(self,value,bounds,uid,subset_by,real_idx):
        if value is None:
            self.value,self.bounds,self.uid,self.real_idx = self._load_(subset_by=subset_by)
        if self.uid is None:
            self.uid = np.arange(1,self.shape[0]+1,dtype=int)
        if self.real_idx is None:
            self.real_idx = np.arange(0,self.shape[0],dtype=int)
            
            
class AbstractVectorDimension(object):
    __metaclass__ = ABCMeta
    
    @property
    def resolution(self):
        ret = np.abs(np.ediff1d(self.value).mean())
        return(ret)
    
    @property
    def shape(self):
        return(self.value.shape)
    
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

            lidx = self.bounds[:,upper_col] >= lower
            uidx = self.bounds[:,lower_col] <= upper
            idx = np.logical_and(lidx,uidx)
            bounds = self.bounds[idx,:]
        
        ret = self.__class__(gi=self.gi,value=self.value[idx],bounds=bounds,
                             uid=self.uid[idx],real_idx=self.real_idx[idx])
        return(ret)


class AbstractLevelDimension(AbstractVectorDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractTemporalDimension(AbstractVectorDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractRowDimension(AbstractVectorDimension,AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractColumnDimension(AbstractRowDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractSpatialDimension(AbstractInterfaceDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractSpatialGrid(AbstractSpatialDimension):
    __metaclass__ = ABCMeta
    
    
class AbstractSpatialVector(AbstractSpatialDimension):
    __metaclass__ = ABCMeta