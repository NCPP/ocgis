import itertools
from ocgis.util.helpers import iter_array


class BaseIterator(object):
    
    def __init__(self,coll):
        self.coll = coll
        
    def __iter__(self):
        raise(NotImplementedError)
    
    def iter_rows(self,headers):
        for row in self:
            yield([row[h] for h in headers],row['geom'])
    
    
class RawIterator(BaseIterator):
    
    def _get_idx_iterator_(self):
        it_tidx = iter_array(self.coll.tid)
        it_gidx = iter_array(self.coll.gid)
        it_variables = self.coll.variables.keys()
        it_idx = itertools.product(it_tidx,it_gidx,it_variables)
        return(it_idx)
    
    def __iter__(self):
        it_idx = self._get_idx_iterator_()
        
        for ((tidx,),gidx,var_key) in it_idx:
            ref = self.coll.variables[var_key]
            for lidx in iter_array(ref.lid):
                vlid = self.coll.vlid.get(ref.lid[lidx],ref.levelvec[lidx])
                ret = dict(
                 tid=self.coll.tid[tidx],
                 gid=self.coll.gid[gidx],
                 geom=self.coll.geom[gidx],
                 time=self.coll.timevec[tidx],
                 lid=ref.lid[lidx],
                 level=ref.levelvec[lidx],
                 vlid=vlid,
#                 vlid=ref.vlid[lidx],
                 value=ref.raw_value[tidx][lidx][gidx],
                 vid=ref.vid,
                 variable=ref.name,
                 ugid=self.coll.geom_dict['ugid']
                           )
                yield(ret)
                
                
class AggIterator(RawIterator):
    
    def __iter__(self):
        it_idx = self._get_idx_iterator_()
        
        for ((tidx,),gidx,var_key) in it_idx:
            ref = self.coll.variables[var_key]
            for lidx in iter_array(ref.lid):
                vlid = self.coll.vlid.get(ref.lid[lidx],ref.levelvec[lidx])
                ret = dict(
                 tid=self.coll.tid[tidx],
                 gid=self.coll.gid[gidx],
                 geom=self.coll.geom[gidx],
                 time=self.coll.timevec[tidx],
                 lid=ref.lid[lidx],
                 level=ref.levelvec[lidx],
                 vlid=vlid,
#                 vlid=ref.vlid[lidx],
                 value=ref.agg_value[tidx][lidx][gidx],
                 vid=ref.vid,
                 variable=ref.name,
                 ugid=self.coll.geom_dict['ugid']
                           )
                yield(ret)
                
                
class CalcIterator(BaseIterator):
    
    def _get_idx_iterator_(self):
        it_tidx = iter_array(self.coll.tgid)
        it_gidx = iter_array(self.coll.gid)
        it_variables = self.coll.variables.keys()
        it_idx = itertools.product(it_tidx,it_gidx,it_variables)
        return(it_idx)
    
    def __iter__(self):
        it_idx = self._get_idx_iterator_()
        
        for ((tidx,),gidx,var_key) in it_idx:
            ref = self.coll.variables[var_key]
            for lidx in iter_array(ref.lid):
                for cidx,(key,value) in enumerate(ref.calc_value.iteritems()):
#                    import ipdb;ipdb.set_trace()
                    ret = dict(
                     tgid=self.coll.tgid[tidx],
                     year=self.coll.year[tidx],
                     month=self.coll.month[tidx],
                     day=self.coll.day[tidx],
                     gid=self.coll.gid[gidx],
                     geom=self.coll.geom[gidx],
                     time=self.coll.timevec[tidx],
                     lid=ref.lid[lidx],
                     level=ref.levelvec[lidx],
#                     vlid=ref.vlid[lidx],
                     vlid=self.coll.vlid.get(ref.lid[lidx],ref.levelvec[lidx]),
                     value=value[tidx][lidx][gidx],
                     vid=ref.vid,
                     variable=ref.name,
                     calc_name=key,
#                     cid=ref.cid[cidx],
                     cid=self.coll.cid.get(key),
                     ugid=self.coll.geom_dict['ugid']
                               )
                    yield(ret)
                    
                    
class RawKeyedIterator(BaseIterator):
    _value_attr = 'raw_value'
    
    def get_iters(self):
        
        def user_geometry():
            yield([self.coll.geom_dict['ugid']])
        
        def geometry():
            for gidx in iter_array(self.coll.gid):
                yield([self.coll.gid[gidx]])
        
        time_headers = ['TID','TIME']
        def time():
            for tidx in iter_array(self.coll.tid):
                yield(self.coll.tid[tidx],self.coll.timevec[tidx])
                    
        def variable():
            for value in self.coll.variables.itervalues():
                yield(value.vid,value.name)
                
        def level():
            for vlid,lid,level_value in self.coll.vlid.iteritems():
                yield(vlid,lid,level_value)
        
        def value():
            for tidx in iter_array(self.coll.tid):
                for gidx in iter_array(self.coll.gid):
                    for value in self.coll.variables.itervalues():
                        vref = getattr(value,self._value_attr)
                        for lidx in iter_array(value.lid):
                            yield(self.coll.geom_dict['ugid'],
                                  self.coll.gid[gidx],
                                  self.coll.tid[tidx],
                                  value.vid,
                                  self.coll.vlid.get(value.lid[lidx],value.levelvec[lidx]),
#                                  value.vlid[lidx],
                                  vref[tidx][lidx][gidx])
        
        ret = {
         'ugid':{'it':user_geometry,'headers':['UGID']},
         'gid':{'it':geometry,'headers':['GID']},
         'tid':{'it':time,'headers':time_headers},
         'vid':{'it':variable,'headers':['VID','VAR_NAME']},
         'vlid':{'it':level,'headers':['VLID','LID','LEVEL']},
         'value':{'it':value,'headers':['UGID','GID','TID','VID','VLID','VALUE']}
               }
        return(ret)
    
    
class AggKeyedIterator(RawKeyedIterator):
    _value_attr = 'agg_value'
    
    
class CalcKeyedIterator(BaseIterator):
    
    def get_iters(self):
        
        def user_geometry():
            yield([self.coll.geom_dict['ugid']])
        
        def geometry():
            for gidx in iter_array(self.coll.gid):
                yield([self.coll.gid[gidx]])
        
        time_headers = ['TGID','YEAR','MONTH','DAY']
        def time():
            for tidx in iter_array(self.coll.tgid):
                tidx = tidx[0]
                yield(self.coll.tgid[tidx],self.coll.year[tidx],
                      self.coll.month[tidx],self.coll.day[tidx])
                    
        def variable():
            for value in self.coll.variables.itervalues():
                yield(value.vid,value.name)
                
        def level():
            for vlid,lid,level_value in self.coll.vlid.iteritems():
                yield(vlid,lid,level_value)
#            for value in self.coll.variables.itervalues():
#                for lidx in iter_array(value.lid):
#                    yield(value.vlid[lidx],value.lid[lidx],value.levelvec[lidx])
        
        def calc():
            for key,value in self.coll.cid.iteritems():
                yield(value,key)
#            for value in self.coll.variables.itervalues():
#                for cidx,key in enumerate(value.calc_value.keys()):
#                    yield(value.cid[cidx],key)
#                break
        
        def value():
            for tidx in iter_array(self.coll.tgid):
                for gidx in iter_array(self.coll.gid):
                    for value in self.coll.variables.itervalues():
                        for lidx in iter_array(value.lid):
                            for calc_name,calc_value in value.calc_value.iteritems():
                                yield(self.coll.geom_dict['ugid'],
                                      self.coll.gid[gidx],
                                      self.coll.tid[tidx],
                                      value.vid,
                                      self.coll.vlid.get(value.lid[lidx],value.levelvec[lidx]),
#                                      value.vlid[lidx],
                                      self.coll.cid[calc_name],
                                      calc_value[tidx][lidx][gidx])
        
        ret = {
         'ugid':{'it':user_geometry,'headers':['UGID']},
         'gid':{'it':geometry,'headers':['GID']},
         'tgid':{'it':time,'headers':time_headers},
         'vid':{'it':variable,'headers':['VID','VAR_NAME']},
         'vlid':{'it':level,'headers':['VLID','LID','LEVEL']},
         'cid':{'it':calc,'headers':['CID','CALC_NAME']},
         'value':{'it':value,'headers':['UGID','GID','TGID','VID','VLID','CID','VALUE']}
               }
        return(ret)
    
    
class MultiKeyedIterator(BaseIterator):
    
    def get_iters(self):
        
        def user_geometry():
            yield([self.coll.geom_dict['ugid']])
        
        def geometry():
            for gidx in iter_array(self.coll.gid):
                yield([self.coll.gid[gidx]])
        
        time_headers = ['TGID','YEAR','MONTH','DAY']
        def time():
            for tidx in iter_array(self.coll.tgid):
                tidx = tidx[0]
                yield(self.coll.tgid[tidx],self.coll.year[tidx],
                      self.coll.month[tidx],self.coll.day[tidx])
                    
#        def variable():
#            for value in self.coll.variables.itervalues():
#                yield(value.vid,value.name)
                
        def level():
            for vlid,lid,level_value in self.coll.vlid.iteritems():
                yield(vlid,lid,level_value)
#            for value in self.coll.variables.itervalues():
#                for lidx in iter_array(value.lid):
#                    yield(value.vlid[lidx],value.lid[lidx],value.levelvec[lidx])
#                break
        
        def multi(): ##cid,key
#            for ii,key in enumerate(self.coll.calc_multi.keys(),start=1):
#                yield(ii,key)
            for key,value in self.coll.cid.iteritems():
                yield(value,key)
        
        def value():
            arch = self.coll.variables[self.coll.variables.keys()[0]]
            for tidx in iter_array(self.coll.tgid):
                tidx = tidx[0]
                for gidx in iter_array(self.coll.gid):
                    for multi_name,multi_value in self.coll.calc_multi.iteritems():
#                    for cidx,value in enumerate(self.coll.calc_multi.itervalues(),start=1):
                        for lidx in range(multi_value.shape[1]):
                            v = multi_value[tidx][lidx][gidx]
                            try:
                                if v.mask:
                                    v = None
                            except AttributeError:
                                pass
                            yield(self.coll.geom_dict['ugid'],
                                  self.coll.gid[gidx],
                                  self.coll.tgid[tidx],
                                  self.coll.vlid.get(arch.lid[lidx],arch.levelvec[lidx]),
                                  self.coll.cid[multi_name],
                                  v)
                    for var_name,variable in self.coll.variables.iteritems():
                        for calc_name,calc_value in variable.calc_value.iteritems():
                            for lidx in iter_array(variable.lid):
                                v = calc_value[tidx][lidx][gidx]
                                yield(self.coll.geom_dict['ugid'],
                                      self.coll.gid[gidx],
                                      self.coll.tgid[tidx],
                                      self.coll.vlid.get(variable.lid[lidx],variable.levelvec[lidx]),
                                      self.coll.cid[calc_name],
                                      v)
        
        ret = {
         'ugid':{'it':user_geometry,'headers':['UGID']},
         'gid':{'it':geometry,'headers':['GID']},
         'tgid':{'it':time,'headers':time_headers},
#         'vid':{'it':variable,'headers':['VID','VAR_NAME']},
         'vlid':{'it':level,'headers':['VLID','LID','LEVEL']},
         'cid':{'it':multi,'headers':['CID','CALC_NAME']},
         'value':{'it':value,'headers':['UGID','GID','TGID','VLID','CID','VALUE']}
               }
        return(ret)