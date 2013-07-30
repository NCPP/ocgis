import re
import json
from ocgis.util.helpers import itersubclasses
import groups
import numpy as np
import itertools
import abc
from ocgis.calc.groups import OcgFunctionGroup
from ocgis.util.logging_ocgis import ocgis_lh


class OcgFunctionTree(object):
    Groups = [groups.BasicStatistics,groups.Thresholds]
    
    @staticmethod
    def get_potentials():
        import library
        
        potentials = []
        for Sc in itersubclasses(OcgFunction):
            if Sc not in [OcgArgFunction,OcgCvArgFunction]:
                sc = Sc()
                potentials.append((sc.name,sc.text))
        return(potentials)
    
    def get_function_list_restful(self,functions):
        funcs = []
        for f in functions:
            fname = re.search('([A-Za-z_]+)',f).group(1)
            obj = self.find(fname)()
            try:
                args = re.search('\(([\d,aggraw]+)\)',f).group(1)
            except AttributeError:
                args = None
            attrs = {'function':getattr(obj,'calculate')}
            raw = False
            if args is not None:
                args = [str(arg) for arg in args.split(',')]
                args_conv = []
                for arg in args:
                    if arg == 'raw':
                        raw = True
                    elif arg == 'agg':
                        raw = False
                    else:
                        args_conv.append(float(arg))
                attrs.update({'args':args_conv})
            attrs.update({'raw':raw})
            if ':' in f:
                name = f.split(':')[1]
            else:
                name = obj.name
            attrs.update(dict(name=name,desc=obj.description))
            funcs.append(attrs)
        return(funcs)
        
    def json(self):
        children = [Group().format() for Group in self.Groups]
        return(json.dumps(children))
    
    def find(self,name):
        for Group in self.Groups:
            for Child in Group().Children:
                if Child().name == name:
                    return(Child)
        raise(AttributeError('function name "{0}" not found'.format(name)))


class OcgFunction(object):
    '''
    Required class attributes to overload:
    
    * **description** (str): A arbitrary length string describing the calculation.
    * **Group** (:class:`ocgis.calc.groups.OcgFunctionGroup`): The calculation group this function belongs to.
    * **dtype** (type): The output data type for this function. Use 32-bit when possible to avoid conversion issues (e.g. netCDF-3).
    
    Optional class attributes to overload:
    
    * **name** (str): The name of the calculation. No spaces or ambiguous characters! If not overloaded, the name defaults to a lowered string version of the class name.
    
    :param values: An array with dimensions of (time,level,row,column) containing the target values.
    :type values: numpy.ma.MaskedArray
    :param groups: A sequence of boolean arrays with individual array dimensions matching the `time` dimension of `values`.
    :type groups: sequence
    :param agg: If True, calculation is performed on raw values from an aggregated data request. This requires the execution of :func:`ocgis.calc.base.OcgFunction.aggregate_spatial` to aggregate the calculations on individual data cells.
    :type agg: bool
    :param weights: Array of weights with dimension (row,column).
    :type weights: numpy.array or numpy.ma.MaskedArray
    :param kwds: Optional keyword parameters to pass to the calculation function.
    :type kwds: dict
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def description(self): str
    @abc.abstractproperty
    def Group(self): OcgFunctionGroup
    @abc.abstractproperty
    def dtype(self): type
    
    nargs = 0
    name = None
    
    def __init__(self,values=None,groups=None,agg=False,weights=None,kwds={}):
        self.values = values
        self.groups = groups
        self.agg = agg
        self.weights = weights
        self.kwds = kwds
        
        self.text = self.__class__.__name__
        if self.name is None:
            self.name = self.text.lower()
    
    def calculate(self):
        '''
        Execute the calculation. The returned array size differs depending on
        the calculation structure.
        
        :rtype: :class:`numpy.ma.MaskedArray`
        '''
        ## holds output from calculation
        fill = self._get_fill_(self.values)
        ## iterate over temporal groups and levels
        for idx,group in enumerate(self.groups):
            value_slice = self.values[group,:,:,:]
            calc = self._calculate_(value_slice,**self.kwds)
            ## we want to leave the mask alone and only fill the data. calculations
            ## are not concerned with the global mask (though they can be).
            fill.data[idx] = calc
        ## if data is calculated on raw values, but area-weighting is required
        ## aggregate the data using provided weights.
        if self.agg:
            ret = self.aggregate_spatial(fill)
        else:
            ret = fill
        return(ret)
    
    def aggregate_spatial(self,fill):
        ##TODO: possible speed-up with array operations
        aw = np.empty((fill.shape[0],fill.shape[1],1,1),dtype=fill.dtype)
        aw = np.ma.array(aw,mask=False)
        for tidx,lidx in itertools.product(range(fill.shape[0]),range(fill.shape[1])):
            aw[tidx,lidx,:] = self._aggregate_spatial_(fill[tidx,lidx,:],self.weights)
        ret = aw
        return(ret)
    
    @abc.abstractmethod
    def _calculate_(self,values,**kwds):
        '''
        The calculation method to overload. Values are explicitly passed to 
        avoid dereferencing. Reducing along the time axis is required (i.e. axis=0).
        
        :param values: Same as :class:`~ocgis.calc.base.OcgFunction` input.
        :param kwds: Any keyword parameters for the function.
        :rtype: numpy.ma.MaskedArray
        '''
        raise(NotImplementedError)
    
    def _aggregate_spatial_(self,values,weights):
        '''
        Optional spatial aggregation method to overload.
        
        :param values: Same as :class:`~ocgis.calc.base.OcgFunction` input.
        :param weights: Same as :class:`~ocgis.calc.base.OcgFunction` input.
        :rtype: numpy.ma.MaskedArray
        '''
        return(np.ma.average(values,weights=weights))
    
    def _get_fill_(self,values,dtype=None):
        new_dtype = dtype or self.dtype
        fill = np.zeros((len(self.groups),values.shape[1],values.shape[2],values.shape[3]),dtype=new_dtype)
        mask = np.zeros(fill.shape,dtype=bool)
        mask[:] = values.mask[0,0,:]
        fill = np.ma.array(fill,mask=mask)
        return(fill)
        
    def format(self):
        raise(NotImplementedError)
#        ret = dict(text=self.text,
#                   checked=self.checked,
#                   leaf=True,
#                   value=self.name,
#                   desc=self.description,
#                   nargs=self.nargs)
#        return(ret)


class OcgArgFunction(OcgFunction):
    '''
    Required class parameter to overload:
    
    * **nargs** (int): Number of required arguments.
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def nargs(self): int
        
        
class OcgCvArgFunction(OcgArgFunction):
    '''
    Required class parameter to overload:
    
    * **keys** ([str,]): These are the aliases of the input data mapping to input variables required by the calculation.
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def keys(self): [str]
    
    def __init__(self,groups=None,agg=False,weights=None,kwds={}):
        super(OcgCvArgFunction,self).__init__(values=kwds,groups=groups,agg=agg,weights=weights,kwds=kwds)
    
    def calculate(self):
        if self.groups is None:
            fill = self._calculate_(**self.kwds)
        else:
            arch = self.kwds[self.keys[0]]
            fill = self._get_fill_(arch)
            ## iterate over temporal groups and levels
            for idx,group in enumerate(self.groups):
                kwds = self._subset_kwds_(group,self.kwds)
                calc = self._calculate_(**kwds)
                calc = self.aggregate_temporal(calc)
                fill[idx] = calc
        if self.agg:
            ret = self.aggregate_spatial(fill)
        else:
            ret = fill
        return(ret)
    
    @abc.abstractmethod
    def _calculate_(self,**kwds):
        '''
        The calculation method to overload. Note the inputs are all keyword arguments.
        
        :rtype: numpy.ma.MaskedArray
        '''
        raise(NotImplementedError)
    
    def aggregate_temporal(self,values):
        return(self._aggregate_temporal_(values))
    
    def _aggregate_temporal_(self,values):
        '''
        Optional temporal aggregation method to overload.
        
        :param values: Same as :class:`~ocgis.calc.base.OcgFunction` input.
        :rtype: numpy.ma.MaskedArray
        '''
        return(np.ma.mean(values,axis=0))
    
    def _subset_kwds_(self,group,kwds):
        ret = {}
        for key,value in kwds.iteritems():
            if key in self.keys:
                u = value[group,:,:,:]
            else:
                u = value
            ret.update({key:u})
        return(ret)


class KeyedFunctionOutput(object):
    '''
    For functions that output data with a different structure.
    '''
    __metaclass__ = abc.ABCMeta
    @abc.abstractproperty
    def output_keys(self):
        '''
        Sequence of output keys returned by the function.
        '''
        [str]
    
    def aggregate_spatial(self,fill):
        exc = NotImplementedError('Spatial aggregation of raw input values not implemented for keyed output functions.')
        ocgis_lh(exc=exc,logger='calc.library')
        
        
class ProtectedFunction(object):
    '''
    For functions that should be run under certain operational conditions.
    '''
    __metaclass__ = abc.ABCMeta
    @classmethod
    @abc.abstractmethod
    def validate(self,ops): pass