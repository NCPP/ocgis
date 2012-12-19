from ocgis.api.dataset.collection.dimension import LevelDimension
from ocgis.api.dataset.collection.collection import Identifier
import numpy as np
from ocgis.util.helpers import iter_array


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
        return(self._iter_rows_(*args,**kwds))
    
    def _iter_rows_(self,*args,**kwds):
        raise(NotImplementedError)


class MeltedIterator(AbstractOcgIterator):
    
    def _get_headers_(self):
        if self.mode == 'raw':
            return(['vid','did','ugid','tid','lid','gid','var_name','uri','time','level','value'])
        elif self.mode == 'calc':
            ret = ['vid','did','ugid','tgid','lid','gid','var_name','uri']
            arch = self.coll.variables[self.coll.variables.keys()[0]]
            ret += arch.temporal_group.groups
            ret += ['level','value']
            return(ret)
    
    def _iter_rows_(self):
        for var in self.coll.variables.itervalues():
            row = {'vid':self.coll.vid.get(var.name),
                   'did':self.coll.did.get(var.uri),
                   'var_name':var.name,
                   'uri':var.uri,
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
            for calc_name,calc_value in var.calc_value.iteritems():
                row.update({'cid':self.coll.cid.get(calc_name),'calc_name':calc_name})
                yield(calc_value,row)
        elif self.mode == 'multi':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError)

    def _iter_time_(self,var):
        if self.mode == 'raw':
            for row in var.temporal:
                yield(row)
        elif self.mode == 'calc':
            for row in var.temporal_group:
                yield(row)
        elif self.mode == 'multi':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError)
        
    def _iter_level_(self,var):
        if var.level is None:
            yield(0,{LevelDimension._name_uid:None,LevelDimension._name_value:None})
        else:
            for lidx,level in var.level:
                yield(lidx,level)
            
#    def _iter_level_(self,var):
#        if var.level is None:
#            yield(0,{'lid':None,LevelDimension._value_name:None})
#        else:
#            coll = self.coll
#            uid_name = 'lid'
#            value_name = var.level._value_name
#            get = coll.lid.get
#            
#            for ii,row in enumerate(var.level.iter_rows(add_bounds=False)):
#                row.update({uid_name:get(row[value_name])})
#                yield(ii,row)
#            
#    def _iter_spatial_(self,var):
#        coll = self.coll
#        uid_name = 'gid'
#        value_name = var.spatial._value_name
#        get = coll.gid.get
#        
#        for ii,row in var.spatial.iter_rows(add_bounds=False,yield_idx=True):
#            row.update({uid_name:get(row[value_name].wkb)})
#            yield(ii,row)


class KeyedIterator(AbstractOcgIterator):
    
    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(*args,**kwds)
        
        self.tid = Identifier(dtype=object)
        self.tgid = Identifier(dtype=object)
        self.lid = Identifier(dtype=object)
        self.gid = Identifier(dtype=object)
        self.ugid = Identifier()
        
    def iter_list(self,*args,**kwds):
        headers = self.get_headers()
        for row in self.iter_rows(*args,**kwds):
            yield([row[h] for h in headers])
        
    def iter_rows(self,coll):
        ## TODO: optimize
        self._add_collection_(coll)
        ugid = coll.ugeom['ugid']
        for var in coll.variables.itervalues():
            did = coll.did.get(var.uri)
            vid = coll.vid.get(var.name)
            for (tidx,lidx,gidx0,gidx1),value in iter_array(var.value,return_value=True):
                tid = self.tid.get(var.temporal.value[tidx])
                lid = self.lid.get(var.level.value[lidx])
                gid = self.gid.get(var.spatial._value[gidx0,gidx1])
                yld = {'did':did,'vid':vid,'tid':tid,'lid':lid,'gid':gid,
                       'value':value,'ugid':ugid}
                yield(yld)
        
    def _add_collection_(self,coll):
        self.ugid.add(coll.ugeom['ugid'],np.array([coll.ugeom['ugid']]))
        for var in coll.variables.itervalues():
            self.tid.add(var.temporal.value,var.temporal.uid)
            if var.temporal_group is not None:
                raise(NotImplementedError)
                self.tgid.add(var.temporal_group.value[:,1],var.temporal_group.uid)
                self.tgid_bounds.add(var.temporal_group.bounds,var.temporal_group.uid)
            self.lid.add(var.level.value,var.level.uid)
            self.gid.add(var.spatial.value.compressed(),var.spatial.uid.compressed())
    
    def get_request_iters(self):
        ret = {}
        identifier = {'did':['uri'],
                      'cid':['calc_name'],
                      'vid':['var_name']}
        for key,value in identifier.iteritems():
            ret.update({key:{'it':self._get_identifier_(key),
                             'headers':[key]+value}})
        return(ret)
            
    def _get_identifier_(self,attr):
        ref = getattr(self.coll,attr)
        storage = ref.storage
        for idx in range(len(ref)):
            yield(storage[idx].tolist())
            
    def _get_headers_(self):
        if self.mode == 'raw':
            return(['vid','did','ugid','tid','lid','gid','value'])
        elif self.mode == 'calc':
            ret = ['vid','did','ugid','tgid','lid','gid',]
            arch = self.coll.variables[self.coll.variables.keys()[0]]
            ret += arch.temporal_group.groups
            ret += ['value']
            return(ret)
        