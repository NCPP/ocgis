import numpy as np
from copy import copy
from collections import OrderedDict
from ocgis.api.interp.iocg.dataset import iterators
from ocgis.util.helpers import iter_array


class OcgVariable(object):
    """Holds variable data. Variables may be climate variables or derived 
    values.
    
    name :: str :: Name of the variable.
    lid :: nd.array :: Level unique identifiers.
    levelvec :: nd.array :: Level descriptions.
    raw_value :: nd.array :: 4-d data array.
    ocg_dataset=None :: OcgDataset
    vid=None :: int :: Unique variable identifier."""
    
    def __init__(self,name,lid,levelvec,raw_value,ocg_dataset=None,vid=None):
        self.name = name
        self.lid = lid
        self.levelvec = levelvec
        self.raw_value = raw_value
        self.ocg_dataset = ocg_dataset
        self.agg_value = None
        self.calc_value = OrderedDict()
        self.vid = vid
        self.vlid = np.array([],dtype=int)
        self.cid = np.array([],dtype=int)


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
                 geom_dict=None):
        self.tid = tid
        self.gid = gid
        self.geom = geom
        self.geom_mask = geom_mask
        self.timevec = timevec
        self.weights = weights
        self.cengine = cengine
        self.geom_dict = geom_dict
        
        self.variables = {}
        self.has_multi = None
        self.calc_multi = OrderedDict()
#        self.calc_value = OrderedDict()
        self._use_agg = None
        self._vlid_inc = 1
        
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
    
#    def _get_ref_(self,mode):
#        '''Depending on the iterator mode different variables are accessed for
#        time and data.
#        
#        mode :: str
#        
#        returns
#        
#        str
#        dict
#        str'''
#        
#        if mode == 'raw':
#            attr = 'raw_value'
#            tid = 'tid'
#        elif mode == 'calc':
#            attr= 'raw_value'
#            tid = 'tgid'
#        elif mode == 'agg':
#            attr = 'agg_value'
#            tid = 'tid'
#        else:
#            raise(NotImplementedError)
#        return(attr,tid)
#        
#    def iter_rows(self,keys,mode='raw'):
#        '''The base iterator method for the collection.
#        
#        keys :: []str :: The keys to pull.
#        mode="raw" :: str :: Iterator mode.
#        
#        yields
#        
#        []<varying>
#        Shapely Polygon or MultiPolygon'''
#        
#        ## the reference attributes
#        attr,tid = self._get_ref_(mode)
#        
#        ## the time data will change if calculations are made.
#        if self.cengine is None:
#            def _time_get_(tidx):
#                return({'tid':self.tid[tidx],'time':self.timevec[tidx]})
#        else:
#            def _time_get_(tidx):
#                return({'tgid':self.tgid[tidx],
#                        'year':self.year[tidx],
#                        'month':self.month[tidx],
#                        'day':self.day[tidx]})
#        
##        ## the level index iterator
##        def _lidx_():
##            for key,value in self.variables.iteritems():
##                for lidx in iter_array(value.lid):
##                    yield(key,lidx)
#        
#        ## get variable data based on index locations
#        def _variable_get_(tidx,gidx,attr):
#            if mode in ['agg','raw']:
#                for value in self.variables.itervalues():
#                    for lidx in iter_array(value.lid):
#                        yield({
#                                'lid':value.lid[lidx],
#                                'level':value.levelvec[lidx],
#                                'vid':value.vid,
#                                'vlid':value.vlid[lidx],
#                                'value':getattr(value,attr)[tidx][lidx][gidx],
#                                'name':value.name
#                                })
#            elif mode == 'calc':
#                for value in self.variables.itervalues():
#                    for lidx in iter_array(value.lid):
#                        for calc_name,calc_value in value.calc_value.iteritems():
#                            yield({
#                                'lid':value.lid[lidx],
#                                'level':value.levelvec[lidx],
#                                'vid':value.vid,
#                                'vlid':value.vlid[lidx],
#                                'value':calc_value[tidx][lidx][gidx],
#                                'name':value.name,
#                                'calc_name':calc_name,
#                                })
#            else:
#                raise(NotImplementedError)
#                
#        
#        ## the nested loop
#        it = itertools.product(
#         iter_array(getattr(self,tid)),
#         iter_array(self.gid),
#         [attr])
#        
#        ## final output loop
#        for ((tidx,),gidx,attr) in it:
#            base = _time_get_(tidx)
#            base.update({'gid':self.gid[gidx],'geom':self.geom[gidx]})
#            for value in _variable_get_(tidx,gidx,attr):
#                base.update(value)
#                ## pull the keys from the dictionary
#                ret = [base[key] for key in keys]
#                yield(ret,base['geom'])
        
    def __repr__(self):
        msg = ('OcgCollection with {n} variable(s): {vars}').\
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
                           mask=self.geom_mask))
    
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
    
    @property
    def _curr_vlid(self):
        '''Increment the current collection-level level identifier.'''
        
        try:
            return(self._vlid_inc)
        finally:
            self._vlid_inc += 1
        
    def _get_value_(self,var_name):
        return(getattr(self.variables[var_name],self._value_attr))
        
    def _iter_items_(self):
        for var_name,value in self.variables.iteritems():
            yield(var_name,getattr(value,self._value_attr))
        
    def add_variable(self,ocg_variable,location='variables'):
        '''Add a variable to the collection.
        
        ocg_variable :: OcgVariable
        location="variables" :: str :: Name of the dictiony to update with new
            variable data.'''
        
        try:
            iterator = iter(ocg_variable)
        except TypeError:
            iterator = iter([ocg_variable])
        for ii in iterator:
            getattr(self,location).update({ii.name:ii})
            for lidx in iter_array(ii.lid):
                ii.vlid = np.append(ii.vlid,self._curr_vlid)
            
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
    
#    def _get_keyed_iterators_(self,mode):
#        '''Iterators used by keyed output.
#        
#        mode :: str
#        
#        returns
#        
#        dict'''
#        
#        attr,tid = self._get_ref_(mode)
#        
#        def geometry():
#            for gidx in iter_array(self.gid):
#                yield([self.gid[gidx]])
#        
#        tattr = getattr(self,tid)    
#        if tid == 'tid':
#            time_headers = ['TID','TIME']
#            def time():
#                for tidx in iter_array(tattr):
#                    yield(tattr[tidx],self.timevec[tidx])
#        else:
#            time_headers = ['TGID','YEAR','MONTH','DAY']
#            def time():
#                for tidx in iter_array(tattr):
#                    tidx = tidx[0]
#                    yield(tattr[tidx],self.year[tidx],
#                          self.month[tidx],self.day[tidx])
#                    
#        def variable():
#            for value in self.variables.itervalues():
#                yield(value.vid,value.name)
#                
#        def level():
#            for value in self.variables.itervalues():
#                for lidx in iter_array(value.lid):
#                    yield(value.vlid[lidx],value.lid[lidx],value.levelvec[lidx])
#        
#        def value():
#            for tidx in iter_array(tattr):
#                tidx = tidx[0]
#                for gidx in iter_array(self.gid):
#                    for value in self.variables.itervalues():
#                        ref = getattr(value,attr)
#                        for lidx in iter_array(value.lid):
#                            yield(self.gid[gidx],tattr[tidx],value.vid,
#                                   value.vlid[lidx],ref[tidx][lidx][gidx])
#        
#        ret = {
#         'geometry':{'it':geometry,'headers':['GID']},
#         'time':{'it':time,'headers':time_headers},
#         'variable':{'it':variable,'headers':['VID','NAME']},
#         'level':{'it':level,'headers':['VLID','LID','LEVEL']},
#         'value':{'it':value,'headers':['GID',tid.upper(),'VID','VLID','VALUE']}
#               }
#        return(ret)