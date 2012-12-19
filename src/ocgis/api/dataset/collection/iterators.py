from ocgis.api.dataset.collection.dimension import LevelDimension


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
        