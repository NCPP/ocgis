import base
import numpy as np
from ocgis.util.spatial import index as si
from itertools import product
from ocgis.util.helpers import make_poly, iter_array
from shapely import prepared
import netCDF4 as nc
from ocgis import constants
from warnings import warn
from abc import ABCMeta, abstractproperty
from ocgis.exc import DummyDimensionEncountered
from ocgis.interface.metadata import NcMetadata
import datetime
from ocgis.interface.projection import get_projection
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union
from shapely.geometry.point import Point
from shapely.geometry.multipoint import MultiPoint


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


class NcRowDimension(NcDimension,base.AbstractRowDimension):
    _name_id = None
    _name_long = None
    axis = 'Y'
    
    
class NcColumnDimension(NcRowDimension,base.AbstractColumnDimension):
    axis = 'X'


class NcTemporalDimension(NcDimension,base.AbstractTemporalDimension):
    axis = 'T'
    _name_id = 'tid'
    _name_long = 'time'
    
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
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        ret = NcDimension._load_.im_func(cls,gi,subset_by=subset_by)
        attrs = gi.metadata['variables'][ret.name]['attrs']
        ret.units = attrs['units']
        ret.calendar = attrs['calendar']
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
                 row=None,column=None,uid=None):
        self.abstraction = abstraction
        super(self.__class__,self).__init__(projection=projection)
        if grid is None:
            self.grid = NcGridDimension(row=row,column=column)
        else:
            self.grid = grid
        if vector is None:
            if self.grid.row.bounds is None or self.abstraction == 'point':
                self.vector = NcPointDimension(grid=self.grid)
            else:
                self.vector = NcPolygonDimension(grid=self.grid)
        else:
            self.vector = vector
    
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
    def weights(self):
        raise(NotImplementedError,'Use "grid" or "vector" weights.')
    
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
        row = self.row[rs.start:rs.stop]
        column = self.column[cs.start:cs.stop]
        uid = self.uid[rs.start:rs.stop,cs.start:cs.stop]
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
    def resolution(self):
        return(np.mean([self.row.resolution,self.column.resolution]))
    
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
        
        ret = self.__class__(grid=vd.grid,geom=geom)
        return(ret)
    
    def intersects(self,polygon):
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
            geom[ii,jj] = test_geom
            if index_intersects(test_geom,index):
                geom_mask[ii,jj] = False
        
        ret = self.__class__(grid=grid,geom=geom,uid=grid.uid)
        return(ret)
    
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


class NcDataset(base.AbstractDataset):
    _dtemporal = NcTemporalDimension
    _dlevel = NcLevelDimension
    _dspatial = NcSpatialDimension

    def __init__(self,*args,**kwds):
        self._abstraction = kwds.pop('abstraction','polygon')
        super(self.__class__,self).__init__(*args,**kwds)
        self.__ds = None
        self.__dim_map = None
        
    def __del__(self):
        try:
            self._ds.close()
        finally:
            pass
        
    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = NcMetadata(self._ds)
        return(self._metadata)
    
    @property
    def value(self):
        if self._value is None:
            ref = self._ds.variables[self.request_dataset.variable]
            row = self.spatial.grid.row.real_idx
            row_start,row_stop = row[0],row[-1]+1
            column = self.spatial.grid.column.real_idx
            column_start,column_stop = column[0],column[-1]+1
            time = self.temporal.real_idx
            time_start,time_stop = time[0],time[-1]+1
            if self.level is None:
                level_start,level_stop = None,None
            else:
                level = self.level.real_idx
                level_start,level_stop = level[0],level[-1]+1
            self._value = self._get_numpy_data_(ref,time_start,time_stop,
             row_start,row_stop,column_start,column_stop,level_start=level_start,
             level_stop=level_stop)
        return(self._value)
    
    def aggregate(self,new_geom_id=1):
        ## will hold the unioned geometry
        new_geometry = np.ones((1,1),dtype=object)
        new_geometry = np.ma.array(new_geometry,mask=False)
        ## get the masked geometries
        geoms = self.spatial.vector.geom.compressed()
        ## store the raw weights
        self.spatial.vector.raw_weights = self.spatial.vector.weights.copy()
        ## break out the MultiPolygon objects. inextricable geometry errors
        ## sometimes occur otherwise
        if self.spatial.abstraction == 'polygon':
            ugeom = []
            for geom in geoms:
                if isinstance(geom,MultiPolygon):
                    for poly in geom:
                        ugeom.append(poly)
                else:
                    ugeom.append(geom)
            ## execute the union
            new_geometry[0,0] = cascaded_union(ugeom)
        elif self.spatial.abstraction == 'point':
            pts = MultiPoint([pt for pt in geoms.flat])
            new_geometry[0,0] = Point(pts.centroid.x,pts.centroid.y)
        else:
            raise(NotImplementedError)
        ## overwrite the original geometry
        self.spatial.vector._geom = new_geometry
        self.spatial.vector.uid = np.ma.array([[new_geom_id]],mask=False)
        ## aggregate the values
        self.raw_value = self.value.copy()
        self._value = self._get_aggregate_sum_()
        self.spatial.vector._weights = None
    
    @property
    def _dim_map(self):
        if self.__dim_map is None:
            self.__dim_map = self._get_dimension_map_()
        return(self.__dim_map)
    
    @property
    def _ds(self):
        if self.__ds is None:
            self.__ds = nc.Dataset(self.request_dataset.uri,'r')
        return(self.__ds)
    
    def get_subset(self,temporal=None,level=None,spatial_operation=None,polygon=None):
        if temporal is not None:
            new_temporal = self.temporal.subset(*temporal)
        else:
            new_temporal = self.temporal
        if level is not None:
            new_level = self.level.subset(*level)
        else:
            new_level = self.level
        if spatial_operation is not None and polygon is not None:
            if spatial_operation == 'intersects':
                new_vector = self.spatial.vector.intersects(polygon)
            elif spatial_operation == 'clip':
                new_vector = self.spatial.vector.clip(polygon)
            new_spatial = NcSpatialDimension(grid=new_vector.grid,vector=new_vector,
                                             projection=self.spatial.projection,
                                             abstraction=self.spatial.abstraction)
        else:
            new_spatial = self.spatial
        ret = self.__class__(request_dataset=self.request_dataset,temporal=new_temporal,
         level=new_level,spatial=new_spatial,metadata=self.metadata,value=None)
        return(ret)
    
    def _get_aggregate_sum_(self):
        value = self.raw_value
        weights = self.spatial.vector.raw_weights

        ## make the output array
        wshape = (value.shape[0],value.shape[1],1,1)
        weighted = np.ma.array(np.empty(wshape,dtype=float),
                                mask=False)
        ## next, weight and sum the data accordingly
        for dim_time in range(value.shape[0]):
            for dim_level in range(value.shape[1]):
                weighted[dim_time,dim_level,0,0] = \
                 np.ma.average(value[dim_time,dim_level,:,:],weights=weights)
        return(weighted)
    
    def _get_axis_(self,dimvar,dims,dim):
        try:
            axis = getattr(dimvar,'axis')
        except AttributeError:
            warn('guessing dimension location with "axis" attribute missing')
            axis = self._guess_by_location_(dims,dim)
        return(axis)
    
    def _get_dimension_map_(self):
        var = self._ds.variables[self.request_dataset.variable]
        dims = var.dimensions
        mp = dict.fromkeys(['T','Z','X','Y'])
        ds = self._ds
        
        ## try to pull dimensions
        for dim in dims:
            try:
                dimvar = ds.variables[dim]
            except KeyError:
                ## search for variable with the matching dimension
                for key,value in self._meta['variables'].iteritems():
                    if len(value['dimensions']) == 1 and value['dimensions'][0] == dim:
                        dimvar = ds.variables[key]
                        break
            axis = self._get_axis_(dimvar,dims,dim)
            mp[axis] = {'variable':dimvar,'dimension':dim}
            
        ## look for bounds variables
        bounds_names = set(constants.name_bounds)
        for key,value in mp.iteritems():
            if value is None:
                continue
            bounds_var = None
            var = value['variable']
            intersection = list(bounds_names.intersection(set(var.ncattrs())))
            try:
                bounds_var = ds.variables[getattr(var,intersection[0])]
            except IndexError:
                warn('no bounds attribute found. searching variable dimensions for bounds information.')
                bounds_names_copy = bounds_names.copy()
                bounds_names_copy.update([value['dimension']])
                for key2,value2 in self._meta['variables'].iteritems():
                    intersection = bounds_names_copy.intersection(set(value2['dimensions']))
                    if len(intersection) == 2:
                        bounds_var = ds.variables[key2]
            value.update({'bounds':bounds_var})
        return(mp)
    
    def _get_numpy_data_(self,variable,time_start,time_stop,row_start,row_stop,
                         column_start,column_stop,level_start=None,level_stop=None):
        if level_start is None:
            npd = variable[time_start:time_stop,row_start:row_stop,
                           column_start:column_stop]
        else:
            npd = variable[time_start:time_stop,level_start:level_stop,
                           row_start:row_stop,column_start:column_stop]
        if not isinstance(npd,np.ma.MaskedArray):
            npd = np.ma.array(npd,mask=False)
        if len(npd.shape) == 3:
            npd = np.ma.expand_dims(npd,1)
        return(npd)
            
    def _guess_by_location_(self,dims,target):
        mp = {3:{0:'T',1:'Y',2:'X'},
              4:{0:'T',2:'Y',3:'X',1:'Z'}}
        return(mp[len(dims)][dims.index(target)])
    
    def _load_axis_(self,kls):
        ref = self._dim_map[kls.axis]
        try:
            value = ref['variable'][:]
        except TypeError:
            raise(DummyDimensionEncountered(kls.axis))
        name = ref['dimension']
        try:
            bounds = ref['bounds'][:]
            name_bounds = ref['bounds']._name
        except Exception as e:
            raise(NotImplementedError)
        ret = kls(value=value,name=name,bounds=bounds,name_bounds=name_bounds)
        return(ret)
