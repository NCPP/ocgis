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
        for var in self.coll.variables.itervalues():
            row = {'vid':self.coll.vid.get(var.name),
                   'did':self.coll.did.get(var.uri),
                   'var_name':var.name,
                   'uri':var.uri,
                   'ugid':self.coll.ugeom['ugid']}
            for value,row in self._iter_value_(var,row):
                for gidx,geom in self._iter_spatial_(var):
                    row.update(geom)
                    for tidx,time in self._iter_time_(var):
                        row.update(time)
                        for lidx,level in self._iter_level_(var):
                            row.update(level)
                            row.update({'value':value[tidx][lidx][gidx]})
                            yield(row)
                        
    def _iter_value_(self,var,row):
        if self.coll._mode == 'raw':
            yield(var.value,row)
        elif self.coll._mode == 'calc':
            for calc_name,calc_value in var.calc_value.iteritems():
                row.update({'cid':self.coll.cid.get(calc_name),'calc_name':calc_name})
                yield(calc_value,row)
        elif self.coll._mode == 'multi':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError)
            
    
    def _iter_time_(self,var):
        if self.coll._mode == 'raw':
            uid_name = self.coll._tid_name
            value_name = var.temporal._value_name
            get = self.coll.tid.get
            for ii,row in enumerate(var.temporal.iter_rows(add_bounds=False)):
                row.update({uid_name:get(row[value_name])})
                yield(ii,row)
        elif self.coll._mode == 'calc':
            get = self.coll.tgid.get
            value = var.temporal.tgdim.value
            for ii,row in var.temporal.tgdim.iter_rows(add_bounds=False,yield_idx=True):
                row.update({'tgid':get(value[ii,:])})
                yield(ii,row)
        elif self.coll._mode == 'multi':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError)
            
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