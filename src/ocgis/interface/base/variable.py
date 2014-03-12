from ocgis.util.logging_ocgis import ocgis_lh
import abc
from collections import OrderedDict
from ocgis.util.helpers import get_iter
import numpy as np
from ocgis import constants
from ocgis.exc import NoUnitsError
from copy import copy


class AbstractValueVariable(object):
    '''
    :param array-like value:
    :param units:
    :type units: str or :class:`cfunits.Units`
    :param :class:`numpy.dtype` dtype:
    :param fill_value:
    :type fill_value: int or float matching type of ``dtype``
    :param str name:
    '''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,value=None,units=None,dtype=None,fill_value=None,name=None):
        self._value = value
        self._dtype = dtype
        self._fill_value = fill_value
        self.name = name
        ## if the units value is not None, then convert to string. cfunits.Units
        ## may be easily handled this way without checking for the module presence.
        self.units = str(units) if units is not None else None
        
    @property
    def cfunits(self):
        ## the cfunits-python module is not a dependency of ocgis and should be
        ## imported on demand
        from cfunits import Units
        return(Units(self.units))
    
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
    def shape(self):
        return(self.value.shape)
    
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
    
    def cfunits_conform(self,to_units,value=None,from_units=None):
        '''
        Conform units of value variable in-place using :mod:`cfunits`.
        
        :param to_units: Target conform units.
        :type t_units: str or :class:`cfunits.Units`
        :param value: Optional value array to use in place of the object's value.
        :type value: :class:`numpy.ma.array`
        :param from_units: Source units to use in place of the object's value.
        :type from_units: str or :class:`cfunits.Units`
        '''
        from cfunits import Units
        ## units are required for conversion
        if self.cfunits == Units(None):
            raise(NoUnitsError)
        ## allow string unit representations to be passed
        if not isinstance(to_units,Units):
            to_units = Units(to_units)
        ## pick the value to convert. this is added to keep the import of the
        ## units library in the AbstractValueVariable.cfunits property
        convert_value = self.value if value is None else value
        ## use the overloaded "from_units" if passed, otherwise use the object-level
        ## attribute
        from_units = self.cfunits if from_units is None else from_units
        ## units are always converted in place. users need to execute their own
        ## deep copies
        self.cfunits.conform(convert_value,from_units,to_units,inplace=True)
        ## update the units attribute with the destination units
        self.units = str(to_units)
        
        return(convert_value)


class AbstractSourcedVariable(AbstractValueVariable):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,data,src_idx=None,value=None,debug=False,did=None,units=None,
                 dtype=None,fill_value=None,name=None):
        if not debug and value is None and data is None:
            ocgis_lh(exc=ValueError('Sourced variables require a data source if no value is passed.'))
        self._data = data
        self._src_idx = src_idx
        self._debug = debug
        self.did = did
        
        AbstractValueVariable.__init__(self,value=value,units=units,dtype=dtype,fill_value=fill_value,
                                       name=name)
        
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
    '''
    :param name: Representative name for the variable.
    :type name: str
    :param alias: Optional unique name for the variable.
    :type alias: str
    :param units: Variable units. If :mod:`cfunits-python` is installed, this will be
     transformed into a :class:`cfunits.Units` object.
    :type units: str
    :param meta: Optional metadata dictionary or object.
    :type meta: dict or object
    :param uid: Optional unique identifier for the Variable.
    :type uid: int
    :param value: Value associated with the variable.
    :type value: np.ndarray
    :param data: Optional data source if no value is passed.
    :type data: object
    :param did: Optional unique identifier for the data source.
    :type did: int
    :param dtype: Optional data type of the object.
    :type dtype: type
    :param fill_value: Option fill value for masked array elements.
    :type fill_value: int or float
    :param conform_units_to: Target units for conversion.
    :type conform_units_to: str convertible to :class:`cfunits.Units`
    '''
    
    def __init__(self,name=None,alias=None,units=None,meta=None,uid=None,
                 value=None,did=None,data=None,debug=False,conform_units_to=None,
                 dtype=None,fill_value=None):
        self.alias = alias or name
        self.meta = meta or {}
        self.uid = uid
        self._conform_units_to = conform_units_to
        
        super(Variable,self).__init__(value=value,data=data,debug=debug,did=did,
                                      units=units,dtype=dtype,fill_value=fill_value,
                                      name=name)
        
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
                ret = np.ma.array(value,mask=False)
            else:
                ret = value
        return(ret)
    
    def _get_value_(self):
        if self._value is None:
            self._set_value_from_source_()
        return(self._value)
    
    def _set_value_from_source_(self):
        ## load the value from source using the referenced field
        self._value = self._field._get_value_from_source_(self._data,self.name)
        ## ensure the new value has the geometry masked applied
        self._field._set_new_value_mask_(self._field,self._field.spatial.get_mask())
        ## if there are units to conform to, execute this now
        if self._conform_units_to:
            self.cfunits_conform(self._conform_units_to,self.value)
    
    
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
