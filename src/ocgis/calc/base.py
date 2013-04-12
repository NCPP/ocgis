import re
import json
from ocgis.util.helpers import itersubclasses
import groups
import numpy as np
import itertools
import abc
from ocgis.calc.groups import OcgFunctionGroup


class OcgFunctionTree(object):
    Groups = [groups.BasicStatistics,groups.Thresholds]
    
    @staticmethod
    def get_potentials():
        """Left in to support HTML query page. Does not support argumented functions."""
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
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def description(self): str
    @abc.abstractproperty
    def Group(self): OcgFunctionGroup
    @abc.abstractproperty
    def dtype(self): type
    
    text = None
    nargs = 0
    checked = False
    name = None
    
    def __init__(self,values=None,groups=None,agg=False,weights=None,kwds={},
                 file_only=False):
        self.values = values
        self.groups = groups
        self.agg = agg
        self.weights = weights
        self.kwds = kwds
        self.file_only = file_only
        
        if self.text is None:
            self.text = self.__class__.__name__
        if self.name is None:
            self.name = self.text.lower()
    
    def calculate(self):
        ## holds output from calculation
        fill = self._get_fill_(self.values)
        if not self.file_only:
            ## iterate over temporal groups and levels
            for idx,group in enumerate(self.groups):
                value_slice = self.values[group,:,:,:]
                calc = self._calculate_(value_slice,**self.kwds)
                fill[idx] = calc
        ## if data is calculated on raw values, but area-weighting is required
        ## aggregate the data using provided weights.
        ret = self.aggregate_spatial(fill)
        return(ret)
    
    def aggregate_spatial(self,fill):
        if self.agg:
            ##TODO: possible speed-up with array operations
            aw = np.empty((fill.shape[0],fill.shape[1],1,1),dtype=fill.dtype)
            aw = np.ma.array(aw,mask=False)
            for tidx,lidx in itertools.product(range(fill.shape[0]),range(fill.shape[1])):
                aw[tidx,lidx,:] = self._aggregate_spatial_(fill[tidx,lidx,:],self.weights)
            ret = aw
        else:
            ret = fill
        return(ret)
    
    @staticmethod
    @abc.abstractmethod
    def _calculate_(values,**kwds):
        raise(NotImplementedError)
    
    @staticmethod
    def _aggregate_spatial_(values,weights):
        return(np.ma.average(values,weights=weights))
    
    def _get_fill_(self,values):
        fill = np.empty((len(self.groups),values.shape[1],values.shape[2],values.shape[3]),dtype=self.dtype)
        mask = np.empty(fill.shape,dtype=bool)
        mask[:] = values.mask[0,0,:]
        fill = np.ma.array(fill,mask=mask)
        return(fill)
        
    def format(self):
        ret = dict(text=self.text,
                   checked=self.checked,
                   leaf=True,
                   value=self.name,
                   desc=self.description,
                   nargs=self.nargs)
        return(ret)


class OcgArgFunction(OcgFunction):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def nargs(self): int
        
        
class OcgCvArgFunction(OcgArgFunction):
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
        ret = self.aggregate_spatial(fill)
        return(ret)
    
    def aggregate_temporal(self,values):
        return(self._aggregate_temporal_(values))
    
    @staticmethod
    def _aggregate_temporal_(values):
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
