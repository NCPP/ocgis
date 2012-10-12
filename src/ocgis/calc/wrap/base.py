import re
import json
from ocgis.util.helpers import itersubclasses
from ocgis.calc.wrap import groups
import numpy as np
from numpy.ma.core import MaskError


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
#                args = [float(a) for a in args.split(',')]
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
        # ret = {'nodes': dict(expanded=True, children=children)}
        return(json.dumps(children))
    
    def find(self,name):
        for Group in self.Groups:
            for Child in Group().Children:
                if Child().name == name:
                    return(Child)
        raise(AttributeError('function name "{0}" not found'.format(name)))


class OcgFunction(object):
    text = None
    description = None
    checked = False
    name = None
    Group = None
    nargs = 0
    dtype = None
    
    def __init__(self,agg=False,raw=False,weights=None):
        self.agg = agg
        self.raw = raw
        self.weights = weights
        
        ## select the axis for the calculation. basically, if the data is
        ## spatially aggregated but the calculation is performed on raw values
        ## it needs to be flattened prior to the calculation.
        if self.agg and self.raw:
            self.axis = None
        else:
            self.axis = 0
        
        if self.text is None:
            self.text = self.__class__.__name__
        if self.name is None:
            self.name = self.text.lower()
        
        for attr in [self.description,self.Group,self.dtype]:
            assert(attr is not None)
    
    def calculate(self,values,out_shape,**kwds):
        
        ret = np.empty(out_shape,dtype=self.dtype)
        for lidx in range(values.shape[1]):
            value_slice = values[:,lidx,:,:]
            assert(len(value_slice.shape) == 3)
            ret[0,lidx,:] = self._calculate_\
                 (value_slice,self.axis,**kwds)
        return(ret)
        
#        if self.axis is None:
#            ret = np.empty(out_shape,dtype=self.dtype)
#            for lidx in range(values.shape[1]):
#                ret[0,lidx,0,0] = self._calculate_\
#                 (values[:,lidx,:,:],self.axis,**kwds)
#        else:
#            ret = self._calculate_(values,self.axis,**kwds).reshape(out_shape)
#        return(ret)
    
    @staticmethod
    def _calculate_(values,axis,**kwds):
        raise(NotImplementedError)
        
    def format(self):
        ret = dict(text=self.text,
                   checked=self.checked,
                   leaf=True,
                   value=self.name,
                   desc=self.description,
                   nargs=self.nargs)
        return(ret)


class OcgArgFunction(OcgFunction):
    nargs = None
    
    def __init__(self,*args,**kwds):
        assert(self.nargs)
        super(OcgArgFunction,self).__init__(*args,**kwds)
        
        
class OcgCvArgFunction(OcgArgFunction):
    keys = None
    
    def __init__(self,*args,**kwds):
        assert(self.keys)
        super(OcgCvArgFunction,self).__init__(*args,**kwds)
        
#    def calculate(self,out_shape,**kwds):
#        ret = self._calculate_(**kwds)
#        if all([self.agg,self.raw]):
#            ret = np.ma.average(ret,weights=self.weights)
#        return(ret.reshape(out_shape))
    
    def calculate(self,out_shape,**kwds):
        if self.axis is None:
            ret = np.empty(out_shape,dtype=self.dtype)
            for lidx in range(out_shape[1]):
                subsetted = self._subset_kwds_(lidx,kwds)
                ret[0,lidx,0,0] = self._run_calculation_(subsetted)
        else:
            ret = self._run_calculation_(kwds).reshape(out_shape)
        return(ret)
    
    def _run_calculation_(self,kwds):
        return(self._spatial_aggregation_(self._calculate_(**kwds)))
    
    def _spatial_aggregation_(self,values):
        if all([self.agg,self.raw]):
            try:
                ret = np.ma.average(values,weights=self.weights)
            except MaskError:
                ret = np.ma.average(values,weights=self.weights[0,:,:])
        else:
            ret = values
        return(ret)
    
    def _subset_kwds_(self,lidx,kwds):
        ret = {}
        for key,value in kwds.iteritems():
            if key in self.keys:
                u = value[:,lidx,:,:]
            else:
                u = value
            ret.update({key:u})
        return(ret)
    
    def _calculate_(self,**kwds):
        raise(NotImplementedError)