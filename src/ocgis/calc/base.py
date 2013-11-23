import numpy as np
import abc
import itertools
from ocgis.interface.base.variable import DerivedVariable, VariableCollection
from ocgis.util.helpers import get_default_or_apply
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis import constants
import logging
from ocgis.exc import SampleSizeNotImplemented, DefinitionValidationError


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
    dtype = None
    Group = None
    @abc.abstractproperty
    def key(self): str
    long_name = ''
    standard_name = ''
    
    def __init__(self,alias=None,dtype=None,field=None,file_only=False,vc=None,
                 parms=None,tgd=None,use_raw_values=False,calc_sample_size=False):
        self.alias = alias or self.key
        self.dtype = dtype or self.dtype
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
    
    def execute(self):
        '''
        Execute the computation over the input field.
        '''
        self._execute_()
        return(self.vc)
    
    def get_function_definition(self):
        ret = {'key':self.key,'alias':self.alias,'parms':self.parms}
        return(ret)
    
    def get_output_units(self,variable):
        '''
        Return the output units of the function.
        '''
        return(None)
    
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
    
    @classmethod
    def validate(self,ops):
        '''
        Optional method to overload that validates the input :class:`ocgis.OcgOperations`.
        '''
        pass
    
    def _add_to_collection_(self,units=None,value=None,parent_variables=None,alias=None):
        ## the value parameters should come in as a dictionary with two keys
        try:
            fill = value['fill']
            sample_size = value['sample_size']
        ## some computations will just pass the array without the sample size
        ## if _get_temporal_agg_fill_ is bypassed.
        except ValueError:
            fill = value
            sample_size = None
        
        units = units or self.get_output_units(parent_variables[0])
        alias = alias or '{0}_{1}'.format(self.alias,parent_variables[0].alias)
        fdef = self.get_function_definition()
        meta = {'attrs':{'standard_name':self.standard_name,
                         'long_name':self.long_name}}
        parents = VariableCollection(variables=parent_variables)
        dv = DerivedVariable(name=self.key,alias=alias,units=units,value=fill,
                             fdef=fdef,parents=parents,meta=meta)
        self.vc.add_variable(dv)
        
        ## add the sample size if it is present in the fill dictionary
        if sample_size is not None:
            meta = {'attrs':{'standard_name':'sample_size',
                             'long_name':'Statistical Sample Size'}}
            dv = DerivedVariable(name=None,alias='n_'+dv.alias,units=None,value=sample_size,
                                 fdef=None,parents=parents,meta=meta)
            self.vc.add_variable(dv)
        
    @abc.abstractmethod
    def _execute_(self): pass
    
    def _format_parms_(self,values):
        return(values)
    
    def _get_parms_(self):
        return(self.parms)
    
    def _get_temporal_agg_fill_(self,value,f=None,parms=None,shp_fill=None):
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
            
            ## subset the values by the current temporal group
            values = value[ir,self._curr_group,il,:,:]
            ## only 3-d data should be sent to the temporal aggregation method
            assert(len(values.shape) == 3)
            ## execute the temporal aggregation or calculation
            cc = f(values,**parms)
            
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
        
class AbstractUnivariateFunction(AbstractFunction):
    '''
    Base class for functions accepting a single univariate input.
    '''
    __metaclass__ = abc.ABCMeta
        
    def _execute_(self):
        for variable in self.field.variables.itervalues():
            fill = self.calculate(variable.value,**self.parms)
            dtype = self.dtype or variable.value.dtype
            if dtype != fill.dtype:
                fill = fill.astype(dtype)
            assert(fill.shape == self.field.shape)
            if self.tgd is not None:
                fill = self._get_temporal_agg_fill_(fill,f=self.aggregate_temporal,parms={})
            else:
                if self.calc_sample_size:
                    msg = 'Sample sizes not relevant for scalar transforms.'
                    ocgis_lh(msg=msg,logger='calc.base',level=logging.WARN)
                fill = self._get_or_pass_spatial_agg_fill_(fill)
            self._add_to_collection_(value=fill,parent_variables=[variable])


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
    
    def aggregate_temporal(self):
        '''
        This operations is always implicit to :meth:`~ocgis.calc.base.AbstractFunction.calculate`.
        '''
        raise(NotImplementedError('aggregation implicit to calculate method'))
    
    def _execute_(self):
        shp_fill = list(self.field.shape)
        shp_fill[1] = len(self.tgd.dgroups)
        for variable in self.field.variables.itervalues():
            
            ## some calculations need information from the current variable iteration
            self._curr_variable = variable
            
            value = self.get_variable_value(variable)
            fill = self._get_temporal_agg_fill_(value,shp_fill=shp_fill)
            self._add_to_collection_(value=fill,parent_variables=[variable])
            
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
    
    def _execute_(self):
        parms = {k:self.get_variable_value(self.field.variables[self.parms[k]]) for k in self.required_variables}
        for k,v in self.parms.iteritems():
            if k not in self.required_variables:
                parms.update({k:v})
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
        units = self.get_output_units()
        self._add_to_collection_(units=units,value=fill,parent_variables=self.field.variables.values(),
                                 alias=self.alias)
        
    def get_output_units(self):
        return('undefined')
    
    @classmethod
    def validate(cls,ops):
        if ops.calc_sample_size:
            from ocgis.api.parms.definition import CalcSampleSize
            exc = DefinitionValidationError(CalcSampleSize,'Multivariate functions do not calculate sample size at this time.')
            ocgis_lh(exc=exc,logger='calc.base')
        
        ## ensure the required variables are presents
        aliases = [d.alias for d in ops.dataset]
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
    
class AbstractKeyedOutputFunction(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def structure_dtype(self): dict
