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
    def uid(self): np.array(dtype=int)
    @abstractproperty
    def _name_id(self): str
    @abstractproperty
    def _name_long(self): str
    
    def __init__(self,gi=None,subset_by=None,value=None,uid=None,bounds=None):
        if value is None and bounds is not None:
            raise(ValueError("Bounds must be passed with an associated value."))
        
        self.gi = None
        self._set_value_bounds_(value,bounds,subset_by)
        self._uid = uid
        
    @abstractmethod
    def _load_(self,subset_by=None):
        return("do some operation here")
    
    def _set_value_bounds_(self,value,bounds,subset_by):
        if value is None:
            self.value, self.bounds = self._load_(subset_by=subset_by)
        else:
            self.value = value
            self.bounds = bounds
            
            
class AbstractVectorDimension(object):
    __metaclass__ = ABCMeta
    
    @property
    def shape(self):
        return(self.value.shape)
    
    @property
    def uid(self):
        if self._uid is None:
            ret = np.arange(1,self.shape[0]+1,dtype=int)
        else:
            ret = self._uid
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