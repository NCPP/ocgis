import base
import numpy as np


class NcLevelDimension(base.AbstractLevelDimension):
    pass


class NcRowDimension(base.AbstractRowDimension):
    _name_id = None
    _name_long = None
    
    @property
    def extent(self):
        if self.bounds is None:
            ret = (self.value.min(),self.value.max())
        else:
            ret = (self.bounds.min(),self.bounds.max())
        return(ret)
    
    def _load_(self,subset_by=None):
        raise(NotImplementedError)
    
    
class NcColumnDimension(base.AbstractColumnDimension,NcRowDimension):
    pass


class NcTemporalDimension(base.AbstractTemporalDimension):
    pass


class NcSpatialDimension(base.AbstractSpatialDimension):
    _name_id = 'gid'
    _name_long = None
    
    def __init__(self,gi=None,subset_by=None,row=None,column=None,uid=None):
        self.row = row
        self.column = column
        self.gi = gi
        if uid is None:
            shp = (self.row.shape[0],self.column.shape[0])
            uid = np.arange(1,(shp[0]*shp[1])+1,dtype=int).reshape(*shp)
            uid = np.ma.array(data=uid,mask=False)
        self.uid = uid
        
    @property
    def extent(self):
        if self.row.bounds is None:
            attr = 'value'
        else:
            attr = 'bounds'
        row = getattr(self.row,attr)
        column = getattr(self.column,attr)
        return(column.min(),row.min(),column.max(),row.max())
    
    @property
    def resolution(self):
        raise(NotImplementedError)
    
    @property
    def shape(self):
        return(self.row.shape[0],self.column.shape[0])
    
    def subset(self,polygon=None):
        if polygon is not None:
            minx,miny,maxx,maxy = polygon.bounds
            row = self.row.subset(miny,maxy)
            column = self.column.subset(minx,maxx)
            uid = self.uid[row.real_idx.min():row.real_idx.max()+1,
                           column.real_idx.min():column.real_idx.max()+1]
            ret = self.__class__(gi=self.gi,row=row,column=column,uid=uid)
        else:
            ret = self
        return(ret)
        
    def _load_(self):
        raise(NotImplementedError)


class NcGlobalInterface(base.AbstractGlobalInterface):
    _dtemporal = NcTemporalDimension
    _dlevel = NcLevelDimension
    _dspatial = NcSpatialDimension
#    _metdata_cls = NcMetadata