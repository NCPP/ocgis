from ocgis.interface import base
from ocgis.interface.nc.dimension import NcTemporalDimension, NcLevelDimension,\
    NcSpatialDimension
from ocgis.interface.metadata import NcMetadata
import numpy as np
import netCDF4 as nc
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.point import Point
from warnings import warn
from ocgis import constants
from ocgis.exc import DummyDimensionEncountered


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
        
    def __getitem__(self,slc):
        if len(slc) == 4:
            raise(NotImplementedError)
        else:
            tidx,rowidx,colidx = slc
            level = None
            _dummy_level = True
        temporal = self.temporal[tidx]
        spatial = self.spatial[rowidx,colidx]
        request_dataset = self.request_dataset
        ret = self.__class__(request_dataset=request_dataset,temporal=temporal,
                             spatial=spatial,level=level)
        ret._dummy_level = _dummy_level
        return(ret)
        
    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = NcMetadata(self._ds)
        return(self._metadata)
    
    @property
    def value(self):
        if self._value is None:
            ref = self._ds.variables[self.request_dataset.variable]
            row_start,row_stop = self._sub_range_(self.spatial.grid.row.real_idx)
            column_start,column_stop = self._sub_range_(self.spatial.grid.column.real_idx)
            time_start,time_stop = self._sub_range_(self.temporal.real_idx)
            if self.level is None:
                level_start,level_stop = None,None
            else:
                level = self.level.real_idx
                level_start,level_stop = level[0],level[-1]+1
            self._value = self._get_numpy_data_(ref,time_start,time_stop,
             row_start,row_stop,column_start,column_stop,level_start=level_start,
             level_stop=level_stop)
            if self.spatial.vector._geom is not None:
                self._value.mask[:,:,:,:] = np.logical_or(self._value.mask[0,:,:,:],self.spatial.vector._geom.mask)
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
    
    def get_iter_value(self,add_bounds=True):
        value = self.value
        
        import ipdb;ipdb.set_trace()
    
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
    
    @staticmethod
    def _sub_range_(arr):
        try:
            ret = (arr[0],arr[-1]+1)
        except IndexError:
            ret = (arr,arr+1)
        return(ret)
