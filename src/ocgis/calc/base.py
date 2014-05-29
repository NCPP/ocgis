import numpy as np
import abc
import itertools
from ocgis.interface.base.variable import DerivedVariable, VariableCollection
from ocgis.util.helpers import get_default_or_apply
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis import constants
import logging
from ocgis.exc import SampleSizeNotImplemented, DefinitionValidationError,\
    UnitsValidationError
from ocgis.util.units import get_are_units_equal_by_string_or_cfunits


class AbstractFunction(object):
    '''
    Required class attributes to overload:
    
    * **description** (str): A arbitrary length string describing the calculation.
    * **dtype** (type): The output data type for this function. Use 32-bit when 
      possible to avoid conversion issues (e.g. netCDF-3). When possible, the input
      data type will be used for the output data type.
    * **key** (str): The function's unique string identifier.
    
    :param alias: The string identifier to use for the calculation.
    :type alias: str
    :param dtype: The output data type.
    :type dtype: np.dtype
    :param field: The field object over which the calculation is applied.
    :type field: :class:`ocgis.interface.base.Field`
    :param file_only: If `True` pass through but compute output sizes, etc.
    :type file_only: bool
    :param vc: The :class:`ocgis.interface.base.variable.VariableCollection` to append
     output calculation arrays to. If `None` a new collection will be created.
    :type vc: :class:`ocgis.interface.base.variable.VariableCollection`
    :param parms: A dictionary of parameter values.
    :type parms: dict
    :param tgd: An instance of :class:`ocgis.interface.base.dimension.temporal.TemporalGroupDimension`.
    :type tgd: :class:`ocgis.interface.base.dimension.temporal.TemporalGroupDimension`
    :param use_raw_values: If `True`, calculation is performed on raw values from an aggregated data request. This requires the execution of :func:`ocgis.calc.base.OcgFunction.aggregate_spatial` to aggregate the calculations on individual data cells.
    :type agg: bool
    :param calc_sample_size: If `True`, also compute sample sizes for the calculation.
    :type calc_sample_size: bool
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def description(self): str
    @abc.abstractproperty
    def dtype(self): None
    Group = None
    @abc.abstractproperty
    def key(self): str
    #: The calculation's long name. Default is the empty string.
    long_name = ''
    #: The calculation's standard name. Default is the empty string.
    standard_name = ''
    #: The calculation's output units. Modify :meth:`get_output_units` for more
    #: complex units calculations. If the units are left as the default '_input'
    #: then the input variable units are maintained. Otherwise, they will be set
    #: to units attribute value. The string flag is used to allow ``None`` units
    #: to be applied.
    units = '_input_'
    
    ## standard empty dictionary to use for calculation outputs when the operation
    ## is file only
    _empty_fill = {'fill':None,'sample_size':None}
    
    def __init__(self,alias=None,dtype=None,field=None,file_only=False,vc=None,
                 parms=None,tgd=None,use_raw_values=False,calc_sample_size=False,
                 fill_value=None):
        self.alias = alias or self.key
        self.dtype = dtype or self.dtype
        self.fill_value = fill_value
        self.vc = vc or VariableCollection()
        self.field = field
        self.file_only = file_only
        self.parms = get_default_or_apply(parms,self._format_parms_,default={})
        self.tgd = tgd
        self.use_raw_values = use_raw_values
        self.calc_sample_size = calc_sample_size
                
    def aggregate_spatial(self,value,weights):
        '''
        Optional method to overload the method for spatial aggregation.
        '''
        ret = np.ma.average(value,weights=weights)
        return(ret)
    
    def aggregate_temporal(self,values,**kwds):
        '''
        Optional method to overload for temporal aggregation.
        '''
        return(np.ma.mean(values,axis=0))
        
    @abc.abstractmethod
    def calculate(self,**kwds):
        '''
        The calculation method to overload. Values are explicitly passed to 
        avoid dereferencing. Reducing along the time axis is required (i.e. axis=0).
        
        :param values: A three-dimensional array with dimensions (time,row,column).
        :type values: :class:`numpy.ma.MaskedArray`
        :param kwds: Any keyword parameters for the function.
        :rtype: `numpy.ma.MaskedArray`
        '''
        pass
    
    def execute(self):
        '''
        Execute the computation over the input field.
        '''
        ## call the subclass execute method
        self._execute_()
        ## allow the field metadata to be modified
        self.set_field_metadata()
        return(self.vc)
    
    def get_function_definition(self):
        ret = {'key':self.key,'alias':self.alias,'parms':self.parms}
        return(ret)
    
    def get_output_units(self,variable):
        '''
        Get the output units.
        '''
        if self.units == '_input_':
            ret = variable.units
        else:
            ret = self.units
        return(ret)
    
    def get_sample_size(self,values):
        to_sum = np.invert(values.mask)
        return(np.ma.sum(to_sum,axis=0))
    
    def get_variable_value(self,variable):
        ## raw values are to be used by the calculation. if this is True, and
        ## no raw field is associated with the input field, then use the standard
        ## value.
        if self.use_raw_values:
            if self.field._raw is None:
                ret = variable.value
            else:
                ret = self.field._raw.variables[variable.alias].value
        else:
            ret = variable.value
        return(ret)
    
    def set_field_metadata(self):
        '''
        Modify the :class:~`ocgis.interface.base.field.Field` metadata dictionary.
        '''
        pass
    
    def set_variable_metadata(self,variable):
        '''
        Set variable level metadata. If units are to be updated, this must be
        done on the "units" attribute of the variable as this value is read 
        directly from the variable object during conversion.
        '''
        pass
    
    @classmethod
    def validate(self,ops):
        '''
        Optional method to overload that validates the input :class:`ocgis.OcgOperations`.
        '''
        pass
    
    def validate_units(self,*args,**kwargs):
        '''
        Optional method to overload for units validation at the calculation level.
        '''
        pass
    
    def _add_to_collection_(self,units=None,value=None,parent_variables=None,alias=None,
                            dtype=None,fill_value=None):
        
        ## dtype should come in with each new variable
        assert(dtype is not None)
        ## if there is no fill value, use the default for the data type
        if fill_value is None:
            fill_value = np.ma.array([],dtype=dtype).fill_value
        
        ## the value parameters should come in as a dictionary with two keys
        try:
            fill = value['fill']
            sample_size = value['sample_size']
        ## some computations will just pass the array without the sample size
        ## if _get_temporal_agg_fill_ is bypassed.
        except ValueError:
            fill = value
            sample_size = None
        
        alias = alias or self.alias
        fdef = self.get_function_definition()
        meta = {'attrs':{'standard_name':self.standard_name,
                         'long_name':self.long_name}}
        parents = VariableCollection(variables=parent_variables)
        
        ## attempt to copy the grid_mapping attribute for the derived variable
        try:
            meta['attrs']['grid_mapping'] = parents.first().meta['attrs']['grid_mapping']
        except KeyError:
            pass
        
        ## if the operation is file only, creating a variable with an empty
        ## value will raise an exception. pass a dummy data source because even
        ## if the value is trying to be loaded it should not be accessible!
        if self.file_only:
            data = 'foo_data_source'
        else:
            data = None
        dv = DerivedVariable(name=self.key,alias=alias,units=units,value=fill,
                             fdef=fdef,parents=parents,meta=meta,data=data,
                             dtype=dtype,fill_value=fill_value)
        
        ## allow more complex manipulations of metadata
        self.set_variable_metadata(dv)
        ## add the variable to the variable collection
        self._set_derived_variable_alias_(dv,parent_variables)
        self.vc.add_variable(dv)
        
        ## add the sample size if it is present in the fill dictionary
        if sample_size is not None:
            meta = {'attrs':{'standard_name':'sample_size',
                             'long_name':'Statistical Sample Size'}}
            dv = DerivedVariable(name=None,alias='n_'+dv.alias,units=None,value=sample_size,
                                 fdef=None,parents=parents,meta=meta,dtype=constants.np_int,
                                 fill_value=fill_value)
            self.vc.add_variable(dv)
        
    @abc.abstractmethod
    def _execute_(self): pass
    
    def _format_parms_(self,values):
        return(values)
    
    def _get_parms_(self):
        return(self.parms)
    
    def _get_slice_and_calculation_(self,f,ir,il,parms,value=None):
        ## subset the values by the current temporal group
        values = value[ir,self._curr_group,il,:,:]
        ## only 3-d data should be sent to the temporal aggregation method
        assert(len(values.shape) == 3)
        ## execute the temporal aggregation or calculation
        cc = f(values,**parms)
        return(cc,values)
    
    def _get_temporal_agg_fill_(self,value=None,f=None,parms=None,shp_fill=None):
        ## if a default data type was provided at initialization, use this value
        ## otherwise use the data type from the input value.
        dtype = self.dtype or value.dtype
        
        ## if no shape is provided for the fill array, create it.
        if shp_fill is None:
            shp_fill = list(self.field.shape)
            shp_fill[1] = len(self.tgd.dgroups)
        fill = np.ma.array(np.zeros(shp_fill,dtype=dtype))
        
        ## this array holds output from the sample size computations
        if self.calc_sample_size:
            fill_sample_size = np.ma.zeros(fill.shape,dtype=constants.np_int)
        else:
            fill_sample_size = None
        
        ## reference the weights if we are using raw values for the calculations
        ## and the data is spatially aggregated.
        if self.use_raw_values and self.field._raw is not None:
            weights = self.field._raw.spatial.weights
        
        ## this is a bit confusing. depending on the computational class we may
        ## be just aggregating temporally or actually executing the calculation.
        f = f or self.calculate
        
        ## choose the constructor parms or those passed to the method directly.
        parms = parms or self.parms
        
        for ir,it,il in itertools.product(*(range(s) for s in fill.shape[0:3])):
            
            ## reference for the current iteration group used by some computations
            self._curr_group = self.tgd.dgroups[it]
            
            cc,values = self._get_slice_and_calculation_(f,ir,il,parms,value=value)
            
            ## compute the sample size of the computation if requested
            if self.calc_sample_size:
                sample_size = self.get_sample_size(values)
                assert(len(sample_size.shape) == 2)
                sample_size = sample_size.reshape(1,1,1,sample_size.shape[0],sample_size.shape[1])
            else:
                sample_size = None
                            
            ## temporal aggregation / calculation should reduce the data to only its spatial
            ## dimensions
            assert(len(cc.shape) == 2)
            ## resize the data back to 5 dimensions
            cc = cc.reshape(1,1,1,cc.shape[0],cc.shape[1])
            
            ## put the data in the fill array
            try:
                fill[ir,it,il,:,:] = cc
                if self.calc_sample_size:
                    fill_sample_size[ir,it,il,:,:] = sample_size
            ## if it doesn't fit, check if we need to spatially aggregate
            except ValueError as e:
                if self.use_raw_values:
                    fill[ir,it,il,:,:] = self.aggregate_spatial(cc,weights)
                    if self.calc_sample_size:
                        fill_sample_size[ir,it,il,:,:] = self.aggregate_spatial(sample_size,weights)
                else:
                    ocgis_lh(exc=e,logger='calc.base')
        
        ## we need to transfer the data mask from the fill to the sample size
        if self.calc_sample_size:
            fill_sample_size.mask = fill.mask.copy()

        return({'fill':fill,'sample_size':fill_sample_size})
    
    def _get_or_pass_spatial_agg_fill_(self,values):
        ## determine if the output data needs to be spatially aggregated
        if self.use_raw_values and values.shape != self.field.shape:
            ret = np.ma.array(np.zeros(self.field.shape,dtype=values.dtype),mask=False,fill_value=values.fill_value)
            weights = self.field._raw.spatial.weights
            r_aggregate_spatial = self.aggregate_spatial
            for ir,it,il in itertools.product(*[range(i) for i in self.field.shape[0:3]]):
                ret[ir,it,il,0,0] = r_aggregate_spatial(values[ir,it,il,:,:],weights=weights)
        else:
            ret = values
        return(ret)
    
    def _set_derived_variable_alias_(self,dv,parent_variables):
        '''
        Set the alias of the derived variable.
        '''
        if len(self.field.variables) > 1:
            original_alias = dv.alias
            dv.alias = '{0}_{1}'.format(dv.alias,parent_variables[0].alias)
            msg = 'Alias updated to maintain uniquencess Changing "{0}" to "{1}".'.format(original_alias,dv.alias)
            ocgis_lh(logger='calc.base',level=logging.WARNING,msg=msg)

        
class AbstractUnivariateFunction(AbstractFunction):
    '''
    Base class for functions accepting a single univariate input.
    '''
    __metaclass__ = abc.ABCMeta
    #: Optional sequence of acceptable string units defintions for input variables.
    #: If this is set to ``None``, no unit validation will occur.
    required_units = None
    
    def validate_units(self,variable):
        if self.required_units is not None:
            matches = [get_are_units_equal_by_string_or_cfunits(variable.units,target,try_cfunits=True) \
                       for target in self.required_units]
            if not any(matches):
                raise(UnitsValidationError(variable,self.required_units,self.key))
        
    def _execute_(self):
        for variable in self.field.variables.itervalues():
            
            self.validate_units(variable)
            
            if self.file_only:
                fill = self._empty_fill
            else:
                fill = self.calculate(variable.value,**self.parms)
                
            dtype = self.dtype or variable.dtype
            if not self.file_only:
                if dtype != fill.dtype:
                    fill = fill.astype(dtype)
                assert(fill.shape == self.field.shape)
            
            if not self.file_only:
                if self.tgd is not None:
                    fill = self._get_temporal_agg_fill_(fill,f=self.aggregate_temporal,parms={})
                else:
                    if self.calc_sample_size:
                        msg = 'Sample sizes not relevant for scalar transforms.'
                        ocgis_lh(msg=msg,logger='calc.base',level=logging.WARN)
                    fill = self._get_or_pass_spatial_agg_fill_(fill)
                    
            units = self.get_output_units(variable)
                                        
            self._add_to_collection_(value=fill,parent_variables=[variable],
                                     dtype=self.dtype,fill_value=self.fill_value,
                                     units=units)


class AbstractParameterizedFunction(AbstractFunction):
    '''
    Base class for functions accepting parameters.
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def parms_definition(self):
        '''
        A dictionary describing the input parameters with keys corresponding to
        parameter names and values to their types. Set the type to `None` for no
        type checking.
        
        >>> {'threshold':float,'operation':str,'basis':None}
        '''
        dict
    
    def _format_parms_(self,values):
        ret = {}
        for k,v in values.iteritems():
            try:
                formatted = self.parms_definition[k](v)
            ## likely a nonetype
            except TypeError as e:
                if self.parms_definition[k] is None:
                    formatted = v
                else:
                    ocgis_lh(exc=e,logger='calc.base')
            ## likely a required variable for a multivariate calculation
            except KeyError as e:
                if k in self.required_variables:
                    formatted = values[k]
                else:
                    ocgis_lh(exc=e,logger='calc.base')
            ret.update({k:formatted})
        return(ret)

        
class AbstractUnivariateSetFunction(AbstractUnivariateFunction):
    '''
    Base class for functions operating on a single variable but always reducing
    intput data along the time dimension.
    '''
    __metaclass__ = abc.ABCMeta
    
    def aggregate_temporal(self,*args,**kwargs):
        '''
        This operations is always implicit to :meth:`~ocgis.calc.base.AbstractFunction.calculate`.
        '''
        raise(NotImplementedError('aggregation implicit to calculate method'))
    
    def _execute_(self):
        shp_fill = list(self.field.shape)
        shp_fill[1] = len(self.tgd.dgroups)
        for variable in self.field.variables.itervalues():
            
            self.validate_units(variable)
            
            if self.file_only:
                fill = self._empty_fill
            else:
                ## some calculations need information from the current variable iteration
                self._curr_variable = variable
                ## return the value from the variable
                value = self.get_variable_value(variable)
                ## execute the calculations
                fill = self._get_temporal_agg_fill_(value,shp_fill=shp_fill)
            ## get the variable's output units
            units = self.get_output_units(variable)
            ## add the output to the variable collection
            self._add_to_collection_(value=fill,parent_variables=[variable],
                                     dtype=self.dtype,fill_value=self.fill_value,
                                     units=units)
            
    @classmethod
    def validate(cls,ops):
        if ops.calc_grouping is None:
            from ocgis.api.parms.definition import Calc
            msg = 'Set functions must have a temporal grouping.'
            ocgis_lh(exc=DefinitionValidationError(Calc,msg),logger='calc.base')
    

class AbstractMultivariateFunction(AbstractFunction):
    '''
    Base class for functions operating on multivariate inputs.
    '''
    __metaclass__ = abc.ABCMeta
    #: Optional dictionary mapping unit definitions for required variables.
    #: For example: required_units = {'tas':'fahrenheit','rhs':'percent'}
    required_units = None
    #: If True, time aggregation is external to the calculation and will require
    #: running the standard time aggregation methods.
    time_aggregation_external = True
    
    def __init__(self,*args,**kwds):
        if kwds.get('calc_sample_size') is True:
            exc = SampleSizeNotImplemented(self.__class__,'Multivariate functions do not calculate sample size at this time.')
            ocgis_lh(exc=exc,logger='calc.base')
        else:
            AbstractFunction.__init__(self,*args,**kwds)
    
    @abc.abstractproperty
    def required_variables(self):
        '''
        Required property/attribute containing the list of input variables expected
        by the function.
        
        >>> ['tas','rhs']
        '''
        [str]
        
    def get_output_units(self,*args,**kwds):
        return(None)
    
    def _get_slice_and_calculation_(self,f,ir,il,parms,value=None):
        if self.time_aggregation_external:
            ret = AbstractFunction._get_slice_and_calculation_(self,f,ir,il,parms,value=value)
        else:
            new_parms = {}
            for k,v in parms.iteritems():
                if k in self.required_variables:
                    new_parms[k] = v[ir,self._curr_group,il,:,:]
                else:
                    new_parms[k] = v
            cc = f(**new_parms)
            ret = (cc,None)
        return(ret)
    
    def _execute_(self):
        
        self.validate_units()
        
        try:
            parms = {k:self.get_variable_value(self.field.variables[self.parms[k]]) for k in self.required_variables}
        ## try again without the parms dictionary
        except KeyError:
            parms = {k:self.get_variable_value(self.field.variables[k]) for k in self.required_variables}
            
        for k,v in self.parms.iteritems():
            if k not in self.required_variables:
                parms.update({k:v})
        
        if self.file_only:
            fill = self._empty_fill
        else:
            if self.time_aggregation_external:
                fill = self.calculate(**parms)
                
                if self.dtype is not None:
                    fill = fill.astype(self.dtype)
                if not self.use_raw_values:
                    assert(fill.shape == self.field.shape)
                else:
                    assert(fill.shape[0:3] == self.field.shape[0:3])
                if self.tgd is not None:
                    fill = self._get_temporal_agg_fill_(fill,f=self.aggregate_temporal,parms={})
                else:
                    fill = self._get_or_pass_spatial_agg_fill_(fill)
            else:
                fill = self._get_temporal_agg_fill_(parms=parms)
                
        units = self.get_output_units()
                        
        self._add_to_collection_(value=fill,parent_variables=self.field.variables.values(),
                                 alias=self.alias,dtype=self.dtype,fill_value=self.fill_value,
                                 units=units)
    
    @classmethod
    def validate(cls,ops):
        if ops.calc_sample_size:
            from ocgis.api.parms.definition import CalcSampleSize
            exc = DefinitionValidationError(CalcSampleSize,'Multivariate functions do not calculate sample size at this time.')
            ocgis_lh(exc=exc,logger='calc.base')
        
        ## ensure the required variables are presents
        aliases = [d.alias for d in ops.dataset.itervalues()]
        should_raise = False
        for c in ops.calc:
            if c['func'] == cls.key:
                if not len(set(c['kwds'].keys()).intersection(set(cls.required_variables))) >= 2:
                    should_raise = True
                if not len(set(c['kwds'].values()).intersection(set(aliases))) >= 2:
                    should_raise = True
                break
        if should_raise:
            from ocgis.api.parms.definition import Calc
            exc = DefinitionValidationError(Calc,'Variable aliases are missing for multivariate function "{0}". Required variable aliases are: {1}.'.format(cls.__name__,cls.required_variables))
            ocgis_lh(exc=exc,logger='calc.base')
            
    def validate_units(self):
        if self.required_units is not None:
            for required_variable in self.required_variables:
                alias_variable = self.parms[required_variable]
                variable = self.field.variables[alias_variable]
                source = variable.units
                target = self.required_units[required_variable]
                match = get_are_units_equal_by_string_or_cfunits(source,target,try_cfunits=True)
                if match == False:
                    raise(UnitsValidationError(variable,target,self.key))
                
    def _set_derived_variable_alias_(self,dv,parent_variables):
        pass

    
class AbstractKeyedOutputFunction(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def structure_dtype(self): dict
