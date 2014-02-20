from ocgis.util.logging_ocgis import ocgis_lh
import abc
from collections import OrderedDict
from ocgis.util.helpers import get_iter
import numpy as np
from ocgis import constants
from copy import copy


class AbstractValueVariable(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,value=None,dtype=None,fill_value=None):
        self._value = value
        self._dtype = dtype
        self._fill_value = fill_value
        
    @property
    def dtype(self):
        if self._dtype is None:
            if self._value is None:
                raise(ValueError('dtype not specified at object initialization and value has not been loaded.'))
            else:
                ret = self.value.dtype
        else:
            ret = self._dtype
        return(ret)
    
    @property
    def fill_value(self):
        if self._fill_value is None:
            if self._value is None:
                raise(ValueError('fill_value not specified at object initialization and value has not been loaded.'))
            else:
                ret = self.value.fill_value
        else:
            ret = self._fill_value
        return(ret)
    
    @property
    def value(self):
        if self._value is None:
            self._value = self._get_value_()
        return(self._value)
    def _get_value_(self):
        return(self._value)
    
    @property
    def _value(self):
        return(self.__value)
    @_value.setter
    def _value(self,value):
        self.__value = self._format_private_value_(value)
    @abc.abstractmethod
    def _format_private_value_(self,value):
        return(value)


class AbstractSourcedVariable(AbstractValueVariable):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,data,src_idx=None,value=None,debug=False,did=None,dtype=None,
                 fill_value=None):
        if not debug and value is None and data is None:
            ocgis_lh(exc=ValueError('Sourced variables require a data source if no value is passed.'))
        self._data = data
        self._src_idx = src_idx
        self._debug = debug
        self.did = did
        
        super(AbstractSourcedVariable,self).__init__(value=value,dtype=dtype,
                                                     fill_value=fill_value)
        
    @property
    def _src_idx(self):
        return(self.__src_idx)
    @_src_idx.setter
    def _src_idx(self,value):
        self.__src_idx = self._format_src_idx_(value)
    
    def _format_src_idx_(self,value):
        if value is None:
            ret = value
        else:
            ret = value
        return(ret)
    
    def _get_value_(self):
        if self._data is None and self._value is None:
            ocgis_lh(exc=ValueError('Values were requested from data source, but no data source is available.'))
        elif self._src_idx is None and self._value is None:
            ocgis_lh(exc=ValueError('Values were requested from data source, but no source index source is available.'))
        else:
            self._set_value_from_source_()
        return(self._value)
            
    @abc.abstractmethod
    def _set_value_from_source_(self): pass


class Variable(AbstractSourcedVariable):
    
    def __init__(self,name=None,alias=None,units=None,meta=None,uid=None,
                 value=None,data=None,debug=False,did=None,fill_value=None,
                 dtype=None):
        self.name = name
        self.alias = alias or name
        self.units = units
        self.meta = meta or {}
        self.uid = uid
        
        super(Variable,self).__init__(value=value,data=data,debug=debug,did=did,
                                      dtype=dtype,fill_value=fill_value)
        
    def __getitem__(self,slc):
        ret = copy(self)
        if ret._value is not None:
            ret._value = self._value[slc]
        return(ret)
                
    def __repr__(self):
        ret = '{0}(alias="{1}",name="{2}",units="{3}")'.format(self.__class__.__name__,self.alias,self.name,self.units)
        return(ret)
    
    def _format_private_value_(self,value):
        if value is None:
            ret = None
        else:
            assert(isinstance(value,np.ndarray))
            if not isinstance(value,np.ma.MaskedArray):
                ret = np.ma.array(value,mask=False,fill_value=constants.fill_value)
            else:
                ret = value
        return(ret)
    
    def _get_value_(self):
        if self._value is None:
            self._set_value_from_source_()
        return(self._value)
    
    def _set_value_from_source_(self):
        self._value = self._field._get_value_from_source_(self._data,self.name)
        self._field._set_new_value_mask_(self._field,self._field.spatial.get_mask())
    
    
class VariableCollection(object):
    
    def __init__(self,**kwds):
        self._uid_ctr = 1
        variables = kwds.pop('variables',None)
        
        self._storage = OrderedDict()
            
        if variables is not None:
            for variable in get_iter(variables,dtype=Variable):
                self.add_variable(variable)
                
    def __getitem__(self,*args,**kwds):
        return(self._storage.__getitem__(*args,**kwds))
    
    def __len__(self):
        return(len(self._storage))
                
    def add_variable(self,variable):
        assert(isinstance(variable,Variable))
        assert(variable.alias not in self._storage)
        if variable.uid is None:
            variable.uid = self._uid_ctr
            self._uid_ctr += 1
        self._storage.update({variable.alias:variable})
        
    def first(self):
        for value in self.itervalues():
            return(value)
        
    def iteritems(self):
        for k,v in self._storage.iteritems():
            yield(k,v)
        
    def itervalues(self):
        for value in self._storage.itervalues():
            yield(value)
            
    def keys(self):
        return(self._storage.keys())
            
    def values(self):
        return(self._storage.values())
        
    def _get_sliced_variables_(self,slc):
        variables = [v.__getitem__(slc) for v in self.itervalues()]
        ret = VariableCollection(variables=variables)
        return(ret)
        
        
class DerivedVariable(Variable):
    
    def __init__(self,**kwds):
        self.fdef = kwds.pop('fdef')
        self.parents = kwds.pop('parents')
        
        super(DerivedVariable,self).__init__(**kwds)
