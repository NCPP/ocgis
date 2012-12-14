from itertools import product
from ocgis.dev.collection.dimension.dimension import LevelDimension


class OcgIterator(object):
    
    def __init__(self,coll):
        self.coll = coll
        
    def iter_rows(self,*args,**kwds):
        return(self._iter_rows_(*args,**kwds))
    
    def _iter_rows_(self,*args,**kwds):
        raise(NotImplementedError)


class MeltedIterator(OcgIterator):
    
    def _iter_rows_(self):
        coll = self.coll
        
        for var in coll.variables.itervalues():
            value = var.value
            row = {'vid':coll.vid.get(var.name),
                   'did':coll.did.get(var.uri),
                   'var_name':var.name,
                   'uri':var.uri,
                   'ugid':coll.ugeom['ugid']}
            for gidx,geom in self._iter_spatial_(var):
                row.update(geom)
                for tidx,time in self._iter_time_(var):
                    row.update(time)
                    for lidx,level in self._iter_level_(var):
                        row.update(level)
                        row.update({'value':value[tidx][lidx][gidx]})
                        yield(row)
    
    def _iter_time_(self,var):
        coll = self.coll
        uid_name = coll._tid_name
        value_name = var.temporal._value_name
        get = coll.tid.get
        
        for ii,row in enumerate(var.temporal.iter_rows(add_bounds=False)):
            row.update({uid_name:get(row[value_name])})
            yield(ii,row)
            
    def _iter_level_(self,var):
        if var.level is None:
            yield(0,{'lid':None,LevelDimension._value_name:None})
        else:
            coll = self.coll
            uid_name = 'lid'
            value_name = var.level._value_name
            get = coll.lid.get
            
            for ii,row in enumerate(var.level.iter_rows(add_bounds=False)):
                row.update({uid_name:get(row[value_name])})
                yield(ii,row)
            
    def _iter_spatial_(self,var):
        coll = self.coll
        uid_name = 'gid'
        value_name = var.spatial._value_name
        get = coll.gid.get
        
        for ii,row in var.spatial.iter_rows(add_bounds=False,yield_idx=True):
            row.update({uid_name:get(row[value_name].wkb)})
            yield(ii,row)


class KeyedIterator(OcgIterator):
    
    def iter_rows(self,add_bounds=True):
        import ipdb;ipdb.set_trace()