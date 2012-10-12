from ocgis.util.helpers import iter_array
from shapely.geometry.point import Point
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkb
import itertools


class OcgRawIterator(object):
    
    def __init__(self,coll,vkey,ocg_dataset,to_sr,cengine=None):
        self.coll = coll
        self.vkey = vkey
        self.ocg_dataset = ocg_dataset
        self.to_sr = to_sr
        self.cengine = cengine
        
        self.value_ref = self.coll[self.vkey]
        self.tid = self.coll['tid']
        self.gid = self.coll['gid']
        self.timevec = self.coll['timevec']
        self.geom = self.coll['geom']
        
    def gen_rows(self):
        for ii in self._get_iterator_():
            for jj in ii:
                yield(jj)
                
    def _get_iterator_(self):
        iterator = itertools.product(
                          self._frng_(self.tid),
                          iter_array(self.gid,use_mask=True),
                          self.value_ref.keys())
        iterator = itertools.imap(self._fvar_,iterator)
        return(iterator)
    
    @staticmethod
    def _frng_(ary):
        return(range(0,len(ary)))
            
    def _fvar_(self,(tidx,gidx,var_name)):
        ref = self.value_ref[var_name]
        vid = ref['vid']
        geom,area_km2 = self._process_geom_(self.coll['geom'][gidx])
        for lidx in self._frng_(ref['lid']):
            yield(
                  [self.coll['tid'][tidx],
                   self.coll['gid'][gidx],
                   vid,
                   ref['lid'][lidx],
                   self.coll['timevec'][tidx],
                   var_name,
                   ref['levelvec'][lidx],
                   self.value_ref[var_name]['value'][tidx][lidx][gidx],
                   area_km2],
                  geom
                  )
                
    def _conv_to_multi_(self,geom):
        if isinstance(geom,Point):
            pass
        else:
            try:
                geom = MultiPolygon(geom)
            except TypeError:
                geom = MultiPolygon([geom])
            except AssertionError:
                wkt = geom.wkb
                geom = wkb.loads(wkt)
        return(geom)
    
    def _process_geom_(self,geom):
        geom = self._conv_to_multi_(geom)
        area_km2 = self.ocg_dataset.i.projection.get_area_km2(self.to_sr,geom)
        return(geom,area_km2)
    
    
class OcgAttrIterator(OcgRawIterator):
    
    def __init__(self,*args,**kwds):
        super(OcgAttrIterator,self).__init__(*args,**kwds)
        
        self.attr_ref = self.coll['attr']
        self.attr_keys = self._get_attr_keys_()
    
    def _get_attr_keys_(self):
        ret = set()
        for value in self.attr_ref.itervalues():
            for attr_name in value.keys():
                ret.update([attr_name])
            break
        return(list(ret))
    
    def _get_level_idx_(self):
        ret = []
        for var_name,value in self.value_ref.iteritems():
            for idx in self._frng_(value['lid']):
                ret.append([var_name,idx])
        return(ret)
    
    def _get_iterator_(self):
        iterator = itertools.product(
                          self._frng_(self.cengine.dtime['tgid']),
                          iter_array(self.gid,use_mask=True),
                          self._get_level_idx_())
        iterator = itertools.imap(self._fvar_,iterator)
        return(iterator)
    
    def _fvar_(self,(tgidx,gidx,(var_name,lidx))):
        ref_value = self.value_ref[var_name]
        ref_attr = self.coll['attr']
        vid = ref_value['vid']
        geom,area_km2 = self._process_geom_(self.coll['geom'][gidx])
        base = [self.cengine.dtime['tgid'][tgidx],
                self.gid[gidx],
                vid,
                ref_value['lid'][lidx],
                self.cengine.dtime['year'][tgidx],
                self.cengine.dtime['month'][tgidx],
                self.cengine.dtime['day'][tgidx],
                var_name,
                ref_value['levelvec'][lidx]]
        for attr_value in ref_attr[var_name].itervalues():
            base.append(attr_value[tgidx][lidx][gidx])
        base.append(area_km2)
        yield(base,geom)