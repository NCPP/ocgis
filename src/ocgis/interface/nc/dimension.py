from ocgis.interface import base
import numpy as np
from ocgis.util.spatial import index as si
from itertools import product
from ocgis.util.helpers import make_poly, iter_array
from shapely import prepared
import netCDF4 as nc
from abc import ABCMeta, abstractproperty
from ocgis.exc import DummyDimensionEncountered
import datetime
from ocgis.interface.projection import get_projection
from shapely.geometry.point import Point
from ocgis.util.spatial.wrap import wrap_geoms


class NcDimension(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def axis(self): str
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        if subset_by is not None:
            raise(NotImplementedError)
        else:
            try:
                ret = gi._load_axis_(cls)
            except DummyDimensionEncountered:
                ret = None
        return(ret)


class NcLevelDimension(NcDimension,base.AbstractLevelDimension):
    axis = 'Z'
    _name_id = 'lid'
    _name_long = 'level'
    
    def subset(self,lower,upper):
        import ipdb;ipdb.set_trace()


class NcRowDimension(NcDimension,base.AbstractRowDimension):
    _name_id = None
    _name_long = None
    axis = 'Y'
    
    
class NcColumnDimension(NcRowDimension,base.AbstractColumnDimension):
    axis = 'X'


class NcTemporalGroupDimension(base.AbstractTemporalGroupDimension):
    _name_id = 'tgid'
    _name_long = 'time_group'


class NcTemporalDimension(NcDimension,base.AbstractTemporalDimension):
    axis = 'T'
    _name_id = 'tid'
    _name_long = 'time'
    _dtemporal_group_dimension = NcTemporalGroupDimension
    
    @property
    def resolution(self):
        diffs = np.array([],dtype=float)
        value = self.value
        for tidx,tval in iter_array(value,return_value=True):
            try:
                diffs = np.append(diffs,
                                np.abs((tval-value[tidx[0]+1]).days))
            except IndexError:
                break
        return(diffs.mean())
    
    def get_nc_time(self,values):
        ret = nc.date2num(values,self.units,calendar=self.calendar)
        return(ret)
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        ret = NcDimension._load_.im_func(cls,gi,subset_by=subset_by)
        attrs = gi.metadata['variables'][ret.name]['attrs']
        ret.units = gi._t_units or attrs['units']
        ret.calendar = gi._t_calendar or attrs['calendar']
        ret.value = nc.num2date(ret.value,ret.units,calendar=ret.calendar)
        cls._to_datetime_(ret.value)
        if ret.bounds is not None:
            ret.bounds = nc.num2date(ret.bounds,ret.units,calendar=ret.calendar)
            cls._to_datetime_(ret.bounds)
        return(ret)
    
    @staticmethod
    def _to_datetime_(arr):
        dt = datetime.datetime
        for idx,t in iter_array(arr,return_value=True):
            arr[idx] = dt(t.year,t.month,t.day,
                          t.hour,t.minute,t.second)


class NcSpatialDimension(base.AbstractSpatialDimension):
    _name_id = 'gid'
    _name_long = None
    
    def __init__(self,grid=None,vector=None,projection=None,abstraction='polygon',
                 row=None,column=None):
        self.abstraction = abstraction
        super(self.__class__,self).__init__(projection=projection)
        if grid is None:
            self.grid = NcGridDimension(row=row,column=column)
        else:
            self.grid = grid
        if vector is None:
            if self.grid.row.bounds is None or self.abstraction == 'point':
                self.vector = NcPointDimension(grid=self.grid,uid=self.grid.uid)
            else:
                self.vector = NcPolygonDimension(grid=self.grid,uid=self.grid.uid)
        else:
            self.vector = vector
            
    def __getitem__(self,slc):
        grid = self.grid[slc]
        ret = self.__class__(grid=grid,projection=self.projection,
                             abstraction=self.abstraction)
        return(ret)
    
    @property
    def is_360(self):
        if self.grid.column.bounds is not None:
            check = self.grid.column.bounds
        else:
            check = self.grid.column.value
        if np.any(check > 180.):
            ret = True
        else:
            ret = False
        return(ret)
    
    @property
    def pm(self):
        if self.grid.column.bounds is None:
            raise(NotImplementedError('Prime meridian only valid for bounded data.'))
        else:
            pm = 0.0
            ref = self.grid.column.bounds
            for idx in range(ref.shape[0]):
                if ref[idx,0] < 0 and ref[idx,1] > 0:
                    pm = ref[idx,0]
                    break
            return(pm)
    
    @property
    def weights(self):
        raise(NotImplementedError,'Use "grid" or "vector" weights.')
    
    def get_iter(self):
        geoms = self.vector.geom
        name_id = self._name_id
        uid = self.vector.uid
        
        ret = {}
        for ii,jj in iter_array(geoms):
            ret[name_id] = uid[ii,jj]
            yield(((ii,jj),geoms[ii,jj],ret))
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        if subset_by is not None:
            raise(NotImplementedError)
        else:
            row = gi._load_axis_(NcRowDimension)
            column = gi._load_axis_(NcColumnDimension)
            projection = get_projection(gi._ds)
            ret = cls(row=row,column=column,projection=projection,
                      abstraction=gi._abstraction)
        return(ret)

class NcGridDimension(base.AbstractSpatialGrid):
    _name_id = None
    _name_long = None
    
    def __init__(self,row=None,column=None,uid=None):
        self.row = row
        self.column = column
        self._weights = None
        if uid is None:
            shp = (self.row.shape[0],self.column.shape[0])
            uid = np.arange(1,(shp[0]*shp[1])+1,dtype=int).reshape(*shp)
            uid = np.ma.array(data=uid,mask=False)
        self.uid = uid
        
    def __getitem__(self,idx):
        rs,cs = idx
        row = self.row[rs]
        column = self.column[cs]
        uid = self.uid[rs,cs]
        ret = self.__class__(row=row,column=column,uid=uid)
        return(ret)
        
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
    def is_bounded(self):
        check = map(lambda x: x is not None,[self.row.bounds,self.column.bounds])
        if sum(check) == 1:
            raise(ValueError('Only one axis is bounded. This should not be possible.'))
        elif all(check):
            ret = True
        else:
            ret = False
        return(ret)
    
    @property
    def resolution(self):
        return(np.mean([self.row.resolution,self.column.resolution]))
    
    @property
    def shape(self):
        return(self.row.shape[0],self.column.shape[0])
    
    def get_iter(self):
        raise(NotImplementedError)
    
    def subset(self,polygon=None):
        if polygon is not None:
            minx,miny,maxx,maxy = polygon.bounds
            row = self.row.subset(miny,maxy)
            column = self.column.subset(minx,maxx)
            uid = self.uid[row.real_idx.min():row.real_idx.max()+1,
                           column.real_idx.min():column.real_idx.max()+1]
            ret = self.__class__(row=row,column=column,uid=uid)
        else:
            ret = self
        return(ret)


class NcPolygonDimension(base.AbstractPolygonDimension):
    
    def __init__(self,grid=None,geom=None,uid=None):
        self._geom = geom
        self._weights = None
        self.grid = grid
        self.uid = uid
        
    @property
    def extent(self):
        raise(NotImplementedError)
    
    @property
    def weights(self):
        if self._weights is None:
            geom = self.geom
            weights = np.ones(geom.shape,dtype=float)
            weights = np.ma.array(weights,mask=geom.mask)
            for ii,jj in iter_array(geom):
                weights[ii,jj] = geom[ii,jj].area
            weights = weights/weights.max()
            self._weights = weights
        return(self._weights)
    
    def clip(self,polygon):
        ## perform an intersects operation first
        vd = self.intersects(polygon)
        ## prepare the geometry for intersection
        prep_igeom = prepared.prep(polygon)
        
        ## loop for the intersection
        geom = vd._geom
        for ii,jj in iter_array(geom):
            ref = geom[ii,jj]
            if not prep_igeom.contains(ref):
                new_geom = polygon.intersection(ref)
                geom[ii,jj] = new_geom
        
        ret = self.__class__(grid=vd.grid,geom=geom,uid=vd.uid)
        return(ret)
    
    def get_iter(self):
        raise(NotImplementedError)
    
    def intersects(self,polygon):
        ## reset the values to ensure mask is properly applied
        self._value = None
        ## reset the weights
        self._weights = None
        ## do the initial grid subset
        grid = self.grid.subset(polygon=polygon)
        
        ## construct the spatial index
        index_grid = si.build_index_grid(30.0,polygon)
        index = si.build_index(polygon,index_grid)
        
        ## the fill arrays
        geom = np.ones(grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=True)
        
        ## loop performing the spatial operation
        row = grid.row.bounds
        col = grid.column.bounds
        index_intersects = si.index_intersects
        geom_mask = geom.mask
        for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
            rref = row[ii,:]
            cref = col[jj,:]
            test_geom = make_poly(rref,cref)
            if index_intersects(test_geom,index):
                geom[ii,jj] = test_geom
                geom_mask[ii,jj] = False

        ret = self.__class__(grid=grid,geom=geom,uid=grid.uid)
        return(ret)
    
    def unwrap(self):
        raise(NotImplementedError)
    
    def wrap(self):
        geom = self.geom
        new_geom = np.ma.array(np.empty(self.geom.shape,dtype=object),mask=geom.mask)
        for ii,jj,fill in wrap_geoms(geom,yield_idx=True):
            new_geom[ii,jj] = fill
        self._geom = new_geom
    
    def _get_all_geoms_(self):
        ## the fill arrays
        geom = np.ones(self.grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=False)
        ## loop performing the spatial operation
        row = self.grid.row.bounds
        col = self.grid.column.bounds
        for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
            rref = row[ii,:]
            cref = col[jj,:]
            geom[ii,jj] = make_poly(rref,cref)
        
        return(geom)
    
    
class NcPointDimension(NcPolygonDimension):

    @property
    def extent(self):
        raise(NotImplementedError)

    @property
    def weights(self):
        return(np.ma.array(self.grid.weights,mask=self.geom.mask))

    def clip(self,polygon):
        return(self.intersects(polygon))
    
    def intersects(self,polygon):
        ## do the initial grid subset
        grid = self.grid.subset(polygon=polygon)
        ## a prepared polygon
        prep_polygon = prepared.prep(polygon)
        ## the fill arrays
        geom = np.ones(grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=True)
        geom_mask = geom.mask
        
        row = grid.row.value
        col = grid.column.value
        for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
            pt = Point(row[ii],col[jj])
            geom[ii,jj] = pt
            if prep_polygon.intersects(pt):
                geom_mask[ii,jj] = False

        ret = self.__class__(grid=grid,geom=geom,uid=grid.uid)
        return(ret)

    def _get_all_geoms_(self):
        ## the fill arrays
        geom = np.ones(self.grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=False)
        ## loop performing the spatial operation
        row = self.grid.row.value
        col = self.grid.column.value
        for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
            geom[ii,jj] = Point(row[ii],col[jj])
        return(geom)