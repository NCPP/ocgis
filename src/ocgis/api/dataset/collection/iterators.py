from ocgis.api.dataset.collection.dimension import LevelDimension
import numpy as np
from ocgis.util.helpers import iter_array
from ocgis.api.dataset.collection.collection import ArrayIdentifier,\
    GeometryIdentifier, OcgMultivariateCalculationVariable, DequeIdentifier
from collections import deque
from ocgis.exc import UniqueIdNotFound
from numpy.ma.core import MaskedConstant


class AbstractOcgIterator(object):
    
    def __init__(self,coll,mode=None):
        self.coll = coll
        if mode is None:
            self.mode = self.coll._mode
        else:
            self.mode = mode
        
    def iter_list(self,*args,**kwds):
        headers = self.get_headers()
        for row in self.iter_rows(*args,**kwds):
            yield([row[h] for h in headers],row['geom'])
        
    def get_headers(self,upper=False):
        headers = self._get_headers_()
        if upper:
            headers = [h.upper() for h in headers]
        return(headers)
    
    def _get_headers_(self):
        raise(NotImplementedError)
        
    def iter_rows(self,*args,**kwds):
        ret = self._iter_rows_(*args,**kwds)
        for row in ret:
            if type(row['value']) == MaskedConstant:
                row['value'] = None
            yield(row)
    
    def _iter_rows_(self,*args,**kwds):
        raise(NotImplementedError)


class MeltedIterator(AbstractOcgIterator):
    
    def _get_headers_(self):
        if self.mode == 'raw':
            return(['vid','did','ugid','tid','lid','gid','var_name','uri','time','level','value'])
        elif self.mode == 'calc':
            ret = ['vid','did','ugid','cid','tgid','lid','gid','var_name','uri','calc_name']
            arch = self.coll.variables[self.coll.variables.keys()[0]]
            ret += arch.temporal_group.groups
            ret += ['level','value']
            return(ret)
        else:
            raise(NotImplementedError)
    
    def _iter_rows_(self):
        for var in self.coll.variables.itervalues():
            try:
                row = {'vid':self.coll.vid.get(var.name),
                       'did':self.coll.did.get(var.uri),
                       'var_name':var.name,
                       'uri':var.uri,
                       'ugid':self.coll.ugeom['ugid']}
            except UniqueIdNotFound:
                row = {'vid':None,
                       'did':None,
                       'var_name':None,
                       'uri':None,
                       'ugid':self.coll.ugeom['ugid']}
            for value,row in self._iter_value_(var,row):
                for gidx,geom in var.spatial:
                    row.update(geom)
                    for tidx,time in self._iter_time_(var):
                        row.update(time)
                        for lidx,level in self._iter_level_(var):
                            row.update(level)
                            row.update({'value':value[tidx][lidx][gidx]})
                            yield(row)
                        
    def _iter_value_(self,var,row):
        if self.mode == 'raw':
            yield(var.value,row)
        elif self.mode == 'calc':
            if type(var) == OcgMultivariateCalculationVariable:
                row.update({'cid':self.coll.cid.get(var.name),'calc_name':var.name})
                yield(var.value,row)
            else:
                for calc_name,calc_value in var.calc_value.iteritems():
                    row.update({'cid':self.coll.cid.get(calc_name),'calc_name':calc_name})
                    yield(calc_value,row)
        else:
            raise(NotImplementedError)

    def _iter_time_(self,var):
        if self.mode == 'raw':
            for row in var.temporal:
                yield(row)
        elif self.mode == 'calc':
            if var.temporal_group is None:
                for row in var.temporal:
                    yield(row)
            else:
                for row in var.temporal_group:
                    yield(row)
        else:
            raise(NotImplementedError)
        
    def _iter_level_(self,var):
        if var.level is None:
            yield(0,{LevelDimension._name_uid:None,LevelDimension._name_value:None})
        else:
            for lidx,level in var.level:
                yield(lidx,level)


class KeyedIterator(AbstractOcgIterator):
    
    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(*args,**kwds)
        
#        self.tid = ArrayIdentifier(3)
#        self.tgid = ArrayIdentifier(9)
        self.tid = DequeIdentifier()
        self.tgid = DequeIdentifier()
        self.lid = ArrayIdentifier(3)
        self.gid = GeometryIdentifier()
        self.ugid = deque()
#        self.ugid_gid = Identifier(int,2)
        
    def iter_list(self,*args,**kwds):
        headers = self.get_headers()
        for row in self.iter_rows(*args,**kwds):
            yield([row[h] for h in headers])
        
    def _iter_rows_(self,coll):
        ## TODO: optimize
        self._add_collection_(coll)
        ugid = coll.ugeom['ugid']
        for var in coll.variables.itervalues():
            try:
                did = coll.did.get(var.uri)
                vid = coll.vid.get(var.name)
            except UniqueIdNotFound:
                did = None
                vid = None
            for tidx,lidx,gidx0,gidx1,value,calc_name,tgid in self._iter_value_(var):       
                if calc_name is not None:
                    cid = coll.cid.get(calc_name)
                    tid = None
                else:
                    cid = None
                    tid = var.temporal.uid[tidx]
#                    tid = self.tid.get(var.temporal.value[tidx])
                lid = self.lid.get(var.level.value[lidx])
                gid = var.spatial.uid[gidx0,gidx1]
#                gid = self.gid.get(var.spatial._value[gidx0,gidx1],ugid)
                yld = {'did':did,'vid':vid,'tid':tid,'lid':lid,'gid':gid,
                       'value':value,'ugid':ugid,'cid':cid,'tgid':tgid}
                yield(yld)
                
    def _iter_value_(self,var):
        ## TODO: optimize
        if len(var.calc_value) > 0:
            import ipdb;ipdb.set_trace()
            for k,v in var.calc_value.iteritems():
                for (tidx,lidx,gidx0,gidx1),value in iter_array(v,return_value=True):
                    to_get = np.empty((1,var.temporal_group.value.shape[1]+2),dtype=object)
                    to_get[:,0:-2] = var.temporal_group.value[tidx,:]
                    to_get[:,-2:] = var.temporal_group.bounds[tidx,:]
                    tgid = self.tgid.get(to_get)
                    yield(tidx,lidx,gidx0,gidx1,value,k,tgid)
        elif type(var) == OcgMultivariateCalculationVariable:
            import ipdb;ipdb.set_trace()
        else:
            for gidx0,gidx1 in iter_array(var.spatial.value):
                for tidx,lidx in zip(range(var.value.shape[0]),range(var.value.shape[1])):
                    value = var.value[tidx,lidx,gidx0,gidx1]
                    yield(tidx,lidx,gidx0,gidx1,value,None,None)
#            for (tidx,lidx,gidx0,gidx1),value in iter_array(var.value,return_value=True):
#                yield(tidx,lidx,gidx0,gidx1,value,None,None)
        
    def _add_collection_(self,coll):
        self.ugid.append(coll.ugeom['ugid'])
        for var in coll.variables.itervalues():
            if 'tid' in var._use_for_id:
                self.tid.add(var.temporal.value,var.temporal.uid)
            if 'gid' in var._use_for_id:
                self.gid.add(var.spatial.value.compressed(),
                             var.spatial.uid.compressed(),
                             coll.ugeom['ugid'])
#            self.tid.add(var.temporal.value,var.temporal.uid)
            if var.temporal_group is not None:
                import ipdb;ipdb.set_trace()
                init_vals = np.empty((var.temporal_group.value.shape[0],
                                      var.temporal_group.value.shape[1]+2),dtype=object)
                init_vals[:,-2:] = var.temporal_group.bounds
                init_vals[:,0:-2] = var.temporal_group.value
                self.tgid.add(init_vals,var.temporal_group.uid)
            self.lid.add(var.level.value,var.level.uid)
#            if var._use_for_gid:
#                self.gid.add(var.spatial.value.compressed(),
#                             var.spatial.uid.compressed(),
#                             coll.ugeom['ugid'])
    
    def get_request_iters(self):
        ret = {}
        identifier = {'did':['uri'],
                      'cid':['calc_name'],
                      'vid':['var_name']}
        for key,value in identifier.iteritems():
            ret.update({key:{'it':self._get_identifier_(key),
                             'headers':[key]+value}})
        return(ret)
    
    def get_dimension_iters(self):
        ret = {}

        def _time_():
            for row in self.tid:
                yield(row)
        ret.update({'tid':{'it':_time_(),
                           'headers':['tid','time_lower','time','time_upper']}})
        def _time_group_():
            for row in self.tgid:
                yield(row)
        ret.update({'tgid':{'it':_time_group_(),
                           'headers':['tgid','year','month','day','hour',
                                      'minute','second','microsecond',
                                      'time_group_lower','time_group_upper',]}})
        def _level_():
            for row in self.lid:
                yield(row)
        ret.update({'lid':{'it':_level_(),
                           'headers':['lid','level_lower','level','level_upper']}})
        def _spatial_():
            for uid in self.gid.uid:
                yield([uid])
        ret.update({'gid':{'it':_spatial_(),'headers':['gid']}})
        
        def _ugid_():
            for uid in self.ugid:
                yield([uid])
        ret.update({'ugid':{'it':_ugid_(),'headers':['ugid']}})
        
        return(ret)
            
    def _get_identifier_(self,attr):
        ref = getattr(self.coll,attr)
#        storage = ref.storage
        for idx in range(len(ref)):
            yield(ref.uid[idx],ref.value[idx])
#            yield(storage[idx][0],storage[idx][1][0])
            
    def _get_headers_(self):
        return(['vid','did','cid','ugid','tid','tgid','lid','gid','value'])
