from ocgis.api.dataset.collection.collection import ArrayIdentifier
import numpy as np


class EqualSpatialDimensionMapper(object):
    _ncol = 6
    _dtype = float
    _key = 'gid'
    _id_key = '_use_for_id'
    _iface_name = 'spatial'
    
    def __init__(self,datasets):
        self.datasets = datasets
        self.map = {}
        self.id = self.get_identifier()
        self.update()
        self.reduce()
        
    def update(self):
        for ds in self.datasets:
            iface = getattr(ds['ocg_dataset'].i,self._iface_name)
            self.map[ds['alias']] = self.get_id(iface)
        
    def reduce(self):
        for uid in self.id.uid:
            for k,v in self.map.iteritems():
                if v == uid:
                    for ds in self.datasets:
                        if ds['alias'] == k:
                            if not self._id_key in ds:
                                ds[self._id_key] = []
                            ds[self._id_key].append(self._key)
                            break
                    break
        
    def get_identifier(self):
        return(ArrayIdentifier(self._ncol,dtype=self._dtype))
    
    def get_add(self,iface):
        add = self.get_empty()
        add[0,0:4] = iface.extent().bounds
        add[0,4] = iface.resolution
        add[0,5] = iface.count
        return(add)
    
    def get_id(self,iface):
        add = self.get_add(iface)
        self.id.add(add)
        return(self.id.get(add))
        
    def get_empty(self):
        return(np.empty((1,self._ncol),dtype=self._dtype))
    
    
class EqualTemporalDimensionMapper(EqualSpatialDimensionMapper):
    _ncol = 6
    _dtype = object
    _key = 'tid'
    _iface_name = 'temporal'
    
    def get_add(self,iface):
        add = self.get_empty()
        add[0,0] = iface.resolution
        add[0,1] = iface.units
        add[0,2] = iface.calendar
        add[0,3] = iface.tid.shape[0]
        add[0,4:6] = [iface.value.min(),iface.value.max()]
        return(add)