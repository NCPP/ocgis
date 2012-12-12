import numpy as np
from copy import copy
from collections import OrderedDict
from ocgis.util.helpers import iter_array
from ocgis import env
import iterators


class CalcIdentifier(OrderedDict):
    
    def __init__(self,*args,**kwds):
        self._curr = 1
        super(CalcIdentifier,self).__init__(*args,**kwds)
    
    def add(self,key):
        if key not in self:
            self.update({key:self._curr})
            self._curr += 1
            
            
class VariableLevelIdentifier(OrderedDict):
    
    def __init__(self,*args,**kwds):
        self._curr = 1
        self._dtype = None
        super(VariableLevelIdentifier,self).__init__(*args,**kwds)
    
    def add(self,lid,level_value):
        if self._dtype is None:
            self._dtype = type(level_value)
        key = self._key_(lid,level_value)
        if key not in self:
            self.update({key:self._curr})
            self._curr += 1
            
    def get(self,lid,level_value):
        key = self._key_(lid,level_value)
        return(self[key])
    
    def iteritems(self):
        for key,value in super(VariableLevelIdentifier,self).iteritems():
            lid,level_value = key.split('__')
            lid = int(lid)
            level_value = self._dtype(level_value)
            yield(value,lid,level_value)
    
    def _key_(self,lid,level_value):
        key = '{0}__{1}'.format(lid,level_value)
        return(key)
        
        
class OcgVariable(object):
    """Holds variable data. Variables may be climate variables or derived 
    values.
    
    name :: str :: Name of the variable.
    lid :: nd.array :: Level unique identifiers.
    levelvec :: nd.array :: Level descriptions.
    raw_value :: nd.array :: 4-d data array.
    ocg_dataset=None :: OcgDataset
    vid=None :: int :: Unique variable identifier."""
    
    def __init__(self,name,lid,levelvec,raw_value,ocg_dataset=None,vid=None,
                 levelvec_bounds=None):
        self.name = name
        self.lid = lid
        self.levelvec = levelvec
        self.levelvec_bounds = levelvec_bounds
        self.raw_value = raw_value
        self.ocg_dataset = ocg_dataset
        self.agg_value = None
        self.calc_value = OrderedDict()
        self.vid = vid
        
    def __repr__(self):
        msg = '<{0}>:{1}'.format(self.__class__.__name__,self.name)
        return(msg)


class OcgCollection(object):
    '''Data collection with common reference variables for each OcgVariable.
    
    tid :: nd.array :: Unique time identifiers.
    gid :: nd.array :: Unique geometry identifiers.
    geom :: nd.array :: Shapely geometry objects.
    geom_mask :: nd.array :: Boolean geometry mask. Equivalent to gid.mask.
    timevec :: nd.array :: Datetime.datetime objects.
    weights :: nd.array :: Data weights used in the aggregation.
    cengine=None :: OcgCalculationEngine'''
    
    def __init__(self,tid,gid,geom,geom_mask,timevec,weights,cengine=None,
                 geom_dict=None,timevec_bounds=None):
        self.tid = tid
        self.gid = gid
        self.geom = geom
        self.geom_mask = geom_mask
        self.timevec = timevec
        self.timevec_bounds = timevec_bounds
        self.weights = weights
        self.cengine = cengine
        self.geom_dict = geom_dict
        
        self.cid = CalcIdentifier()
        self.vlid = VariableLevelIdentifier()
        
        self.variables = {}
        self.calc_multi = OrderedDict()
        self._use_agg = None
        self._vlid_inc = 1
    
    @property
    def is_empty(self):
        if len(self.gid) == 0:
            ret = True
        else:
            ret = False
        return(ret)
        
    def get_iter(self,mode):
        if mode == 'raw':
            it = iterators.RawIterator(self)
        elif mode == 'agg':
            it = iterators.AggIterator(self)
        elif mode == 'calc':
            it = iterators.CalcIterator(self)
        else:
            raise(NotImplementedError)
        return(it)
        
    def _copy_variable_(self,key,name,value,vid=None):
        '''Return a copy of the variable overload some components. Used when
        generating variables for derived values.
        
        key :: str :: Which variable to copy.
        name :: str :: Name of the new variable.
        value :: nd.array :: New values.
        vid :: int :: New variable unique identifier.
        
        returns
        
        OcgVariable'''
        
        var = self.variables[key]
        if vid is None:
            vid = var.vid
        newvar = OcgVariable(name,var.lid,var.levelvec,value,var.ocg_dataset,
                             vid=vid)
        return(newvar)

    def __repr__(self):
        msg = ('<OcgCollection> with {n} variable(s): {vars}').\
          format(n=len(self.variables.keys()),
                       vars=self.variables.keys())
        return(msg)
    
    @property
    def _value_attr(self):
        if self._use_agg:
            return('agg_value')
        else:
            return('raw_value')
    
    @property
    def geom_masked(self):
        '''Problem with pickling of masked object arrays requires this
           inefficient property.'''
        
        return(np.ma.array(self.geom,
                           mask=self.geom_mask,
                           fill_value=env.FILL_VALUE))
    
    ## TODO: these calls are inefficient
    @property
    def tgid(self):
        return(self._cengine_dtime_attr_('tgid'))
    @property
    def year(self):
        return(self._cengine_dtime_attr_('year'))
    @property
    def month(self):
        return(self._cengine_dtime_attr_('month'))
    @property
    def day(self):
        return(self._cengine_dtime_attr_('day'))
    
    def _cengine_dtime_attr_(self,attr):
        '''Used by calculation engine properties.'''
        
        if self.cengine is None:
            ret = None
        else:
            ret = self.cengine.dtime[attr]
        return(ret)
        
    def _get_value_(self,var_name):
        return(getattr(self.variables[var_name],self._value_attr))
        
    def _iter_items_(self):
        for var_name,value in self.variables.iteritems():
            yield(var_name,getattr(value,self._value_attr))
        
    def add_variable(self,ocg_variable):
        '''Add a variable to the collection.
        
        ocg_variable :: OcgVariable
        '''
        try:
            iterator = iter(ocg_variable)
        except TypeError:
            iterator = iter([ocg_variable])
        for ii in iterator:
            self.variables.update({ii.name:ii})
            for lidx in iter_array(ii.lid):
                self.vlid.add(ii.lid[lidx],ii.levelvec[lidx])
            
    def _get_shape_dict_(self,n,raw=False):
        '''Used during calculation to ensure shapes are properly returned.'''
        
        shapes = {}
        for var_name in self.variables.keys():
            calc_shape = [n,
                          self.variables[var_name].lid.shape[0],
                          self.geom.shape[0],
                          self.geom.shape[1]]
            calc_mask = np.empty(calc_shape,dtype=bool)
            calc_mask[:,:,:,:] = self.geom_mask
            out_shape = copy(calc_shape)
            out_shape[0] = 1
            wshape = copy(calc_shape)
            if raw is False:
                wshape = wshape[1:]
                fweights = np.ma.array(np.ones(wshape,dtype=float))
            else:
                wshape = list(self.weights.shape)
                wshape.insert(0,self.variables[var_name].lid.shape[0])
                fweights = np.ma.array(np.empty(wshape,dtype=float),
                                       mask=np.zeros(wshape,
                                                     dtype=bool))
                fweights.mask[:] = self.geom_mask
            shapes.update({var_name:{'calc_shape':calc_shape,
                                     'calc_mask':calc_mask,
                                     'out_shape':out_shape,
                                     'fweights':fweights}})
        return(shapes)
