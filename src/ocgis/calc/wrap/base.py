import re
import json
from ocgis.util.helpers import itersubclasses
from ocgis.calc.wrap import groups
import numpy as np
from numpy.ma.core import MaskError
import itertools


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


#class OcgFunction(object):
#    text = None
#    description = None
#    checked = False
#    name = None
#    Group = None
#    nargs = 0
#    dtype = None
#    
#    def __init__(self,agg=False,raw=False,weights=None):
#        self.agg = agg
#        self.raw = raw
#        self.weights = weights
#        
#        ## select the axis for the calculation. basically, if the data is
#        ## spatially aggregated but the calculation is performed on raw values
#        ## it needs to be flattened prior to the calculation.
#        if self.agg and self.raw:
#            self.axis = None
#        else:
#            self.axis = 0
#        
#        if self.text is None:
#            self.text = self.__class__.__name__
#        if self.name is None:
#            self.name = self.text.lower()
#        
#        for attr in [self.description,self.Group,self.dtype]:
#            assert(attr is not None)
#    
#    def calculate(self,values,out_shape,**kwds):
#        ret = np.empty(out_shape,dtype=self.dtype)
#        try:
#            ret = np.ma.array(ret,mask=values.mask[0])
#        except MaskError:
#            if self.agg and self.raw:
#                ret = np.ma.array(ret,mask=np.array([False]).reshape(*ret.shape))
#            else:
#                raise
#        for lidx in range(values.shape[1]):
#            value_slice = values[:,lidx,:,:]
#            assert(len(value_slice.shape) == 3)
#            ret[0,lidx,:] = self._calculate_(value_slice,self.axis,**kwds)
#        return(ret)
#        
##        if self.axis is None:
##            ret = np.empty(out_shape,dtype=self.dtype)
##            for lidx in range(values.shape[1]):
##                ret[0,lidx,0,0] = self._calculate_\
##                 (values[:,lidx,:,:],self.axis,**kwds)
##        else:
##            ret = self._calculate_(values,self.axis,**kwds).reshape(out_shape)
##        return(ret)
#    
#    @staticmethod
#    def _calculate_(values,axis,**kwds):
#        raise(NotImplementedError)
#        
#    def format(self):
#        ret = dict(text=self.text,
#                   checked=self.checked,
#                   leaf=True,
#                   value=self.name,
#                   desc=self.description,
#                   nargs=self.nargs)
#        return(ret)

class OcgFunction(object):
    text = None
    description = None
    checked = False
    name = None
    Group = None
    nargs = 0
    dtype = None
    
    def __init__(self,values=None,groups=None,agg=False,weights=None,kwds={}):
        self.values = values
        self.groups = groups
        self.agg = agg
        self.weights = weights
        self.kwds = kwds
        
        if self.text is None:
            self.text = self.__class__.__name__
        if self.name is None:
            self.name = self.text.lower()
        
        for attr in [self.description,self.Group,self.dtype]:
            assert(attr is not None)
    
    def calculate(self):
        ## holds output from calculation
        fill = self._get_fill_(self.values)
        ## iterate over temporal groups and levels
        for idx,group in enumerate(self.groups):
            import ipdb;ipdb.set_trace()
            for lidx in range(self.values.shape[1]):
                value_slice = self.values[group,lidx,:,:]
                assert(len(value_slice.shape) == 3)
                fill[idx,lidx,:] = self._calculate_(value_slice,**self.kwds)
        ## if data is calculated on raw values, but area-weighting is required
        ## aggregate the data using provided weights.
        ret = self.aggregate(fill)
        return(ret)
    
    def aggregate(self,fill):
        if self.agg:
            aw = np.empty((fill.shape[0],fill.shape[1],1,1),dtype=fill.dtype)
            aw = np.ma.array(aw,mask=False)
            for tidx,lidx in itertools.product(range(fill.shape[0]),range(fill.shape[1])):
#                aw[tidx,lidx,:] = np.ma.average(fill[tidx,lidx,:],weights=self.weights)
                aw[tidx,lidx,:] = self._aggregate_(fill[tidx,lidx,:],self.weights)
            ret = aw
        else:
            ret = fill
        return(ret)
    
    @staticmethod
    def _calculate_(values,**kwds):
        raise(NotImplementedError)
    
    @staticmethod
    def _aggregate_(values,weights):
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
    nargs = None
    
    def __init__(self,*args,**kwds):
        assert(self.nargs)
        super(OcgArgFunction,self).__init__(*args,**kwds)
        
        
class OcgCvArgFunction(OcgArgFunction):
    keys = None
    
    def __init__(self,groups=None,agg=False,weights=None,kwds={}):
        assert(self.keys)
        super(OcgCvArgFunction,self).__init__(values=kwds,groups=groups,agg=agg,weights=weights,kwds=kwds)
        
#    def calculate(self,out_shape,**kwds):
#        ret = self._calculate_(**kwds)
#        if all([self.agg,self.raw]):
#            ret = np.ma.average(ret,weights=self.weights)
#        return(ret.reshape(out_shape))
    
    def calculate(self):
        arch = self.kwds[self.keys[0]]
        fill = self._get_fill_(arch)
        ## iterate over temporal groups and levels
        for idx,group in enumerate(self.groups):
            for lidx in range(arch.shape[1]):
                kwds = self._subset_kwds_(group,lidx,self.kwds)
                fill[idx,lidx,:]= self._calculate_(**kwds)
#                import ipdb;ipdb.set_trace()
#                value_slice = self.values[group,lidx,:,:]
#                assert(len(value_slice.shape) == 3)
#                fill[idx,lidx,:] = self._calculate_(value_slice,**self.kwds)
        ret = self.aggregate(fill)
#        import ipdb;ipdb.set_trace()
#        if self.axis is None:
#            ret = np.empty(out_shape,dtype=self.dtype)
#            for lidx in range(out_shape[1]):
#                subsetted = self._subset_kwds_(lidx,kwds)
#                ret[0,lidx,0,0] = self._run_calculation_(subsetted)
#        else:
#            ret = self._run_calculation_(kwds).reshape(out_shape)
        return(ret)
    
#    def _run_calculation_(self,kwds):
#        ## run calculation
#        ret = self._calculate_(**kwds)
#        ## this method is necessary to account for point-based calculations that
#        ## must be spatially weighted when aggregated.
#        ret = self._spatial_aggregation_(ret)
#        return(ret)
    
#    def _spatial_aggregation_(self,values):
#        if all([self.agg,self.raw]):
#            try:
#                ret = np.ma.average(values,weights=self.weights)
#            except MaskError:
#                ret = np.ma.average(values,weights=self.weights[0,:,:])
#        else:
#            ret = values
#        return(ret)
    
    def _subset_kwds_(self,group,lidx,kwds):
        ret = {}
        for key,value in kwds.iteritems():
            if key in self.keys:
                u = value[group,lidx,:,:]
            else:
                u = value
            ret.update({key:u})
        return(ret)
    
#    def _calculate_(self,**kwds):
#        raise(NotImplementedError)