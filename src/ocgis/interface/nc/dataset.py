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
from ocgis.exc import DummyDimensionEncountered, EmptyData
import itertools
from osgeo.ogr import CreateGeometryFromWkb
from ocgis.constants import reference_projection
from shapely.wkb import loads
import ocgis


class NcDataset(base.AbstractDataset):
    _dtemporal = NcTemporalDimension
    _dlevel = NcLevelDimension
    _dspatial = NcSpatialDimension

    def __init__(self,*args,**kwds):
        self._abstraction = kwds.pop('abstraction','polygon')
        self._t_calendar = kwds.pop('t_calendar',None)
        self._t_units = kwds.pop('t_units',None)
        self._s_proj = kwds.pop('s_proj',None)
        super(self.__class__,self).__init__(*args,**kwds)
        self.__ds = None
        self.__dim_map = None
        self._load_slice = {}
        
    def __del__(self):
        try:
            self._ds.close()
        finally:
            pass
        
    def __getitem__(self,slc):
        if self.level is None:
            if len(slc) != 3:
                raise(IndexError('3 slice elements required for 3-d dataset'))
            else:
                tidx,rowidx,colidx = slc
                level = None
                _dummy_level = True
        else:
            if len(slc) != 4:
                raise(IndexError('4 slice elements required for 4-d dataset'))
            else:
                tidx,lidx,rowidx,colidx = slc
                level = self.level[lidx]
                _dummy_level = False
            
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
            
            try:
                row_start,row_stop = self._sub_range_(self.spatial.grid.row.real_idx)
            ## NcGridMatrixDimension correction
            except AttributeError:
                row_start,row_stop = self._sub_range_(self.spatial.grid.real_idx_row.flatten())
            try:
                column_start,column_stop = self._sub_range_(self.spatial.grid.column.real_idx)
            ## NcGridMatrixDimension correction
            except AttributeError:
                column_start,column_stop = self._sub_range_(self.spatial.grid.real_idx_column.flatten())
                
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
    
    def aggregate(self,new_geom_id=1,clip_geom=None):
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
            if geoms.shape[0] == 0:
                raise(EmptyData)
            else:
                ## for point aggregation, use the provided clip geometry if
                ## passed as opposed to the more confusing multipoint centroid.
                ## there may be a need to implement this later so the old code
                ## remains.
                if clip_geom is None:
                    pts = MultiPoint([pt for pt in geoms.flat])
                    new_fill = Point(pts.centroid.x,pts.centroid.y)
                else:
                    new_fill = clip_geom
                new_geometry[0,0] = new_fill
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
            try:
                self.__ds = nc.Dataset(self.request_dataset.uri,'r')
            ## likely multiple uris...
            except TypeError:
                self.__ds = nc.MFDataset(self.request_dataset.uri,'r')
        return(self.__ds)
    
    def get_iter_value(self,add_bounds=True,add_masked=True,value=None,
                       temporal_group=False):        
        ## check if the reference projection is different than the dataset
        if type(self.spatial.projection) != type(reference_projection) and ocgis.env.WRITE_TO_REFERENCE_PROJECTION:
            project = True
            sr = self.spatial.projection.sr
            to_sr = reference_projection.sr
        else:
            project = False
        
        ## if no external value to iterate over is passed, use the internal value
        if value is None:
            value = self.value
        
        ## reference the mask checker
        is_masked = np.ma.is_masked
        ## reference the value name
        _name_value = self._name_value
        ## reference the fill_value
        fill_value = constants.fill_value
        
        ## if iteration over the temporal groups is requested, reference the
        ## appropriate iterator.
        if temporal_group:
            time_iter = self.temporal.group.get_iter
        else:
            time_iter = self.temporal.get_iter
        
        if self.level is None:
            for (ridx,cidx),geom,gret in self.spatial.get_iter():
                if project:
                    geom = CreateGeometryFromWkb(geom.wkb)
                    geom.AssignSpatialReference(sr)
                    geom.TransformTo(to_sr)
                    geom = loads(geom.ExportToWkb())
                for tidx,tret in time_iter(add_bounds=add_bounds):
                    gret.update(tret)
                    gret['lid'] = None
                    gret['level'] = None
                    ref = value[tidx,0,ridx,cidx]
                    masked = is_masked(ref)
                    if add_masked and masked:
                        ref = fill_value
                    elif not add_masked and masked:
                        continue
                    gret[_name_value] = ref
                    yield(geom,gret)
        else:
            for (ridx,cidx),geom,gret in self.spatial.get_iter():
                if project:
                    geom = CreateGeometryFromWkb(geom.wkb)
                    geom.AssignSpatialReference(sr)
                    geom.TransformTo(to_sr)
                    geom = loads(geom.ExportToWkb())
                for lidx,lret in self.level.get_iter(add_bounds=add_bounds):
                    gret.update(lret)
                    for tidx,tret in time_iter(add_bounds=add_bounds):
                        gret.update(tret)
                        ref = value[tidx,lidx,ridx,cidx]
                        masked = is_masked(value)
                        if add_masked and masked:
                            ref = None
                        elif not add_masked and masked:
                            continue
                        gret[_name_value] = ref
                        yield(geom,gret)
    
    def get_subset(self,temporal=None,level=None,spatial_operation=None,igeom=None):
        if temporal is not None:
            new_temporal = self.temporal.subset(*temporal)
        else:
            new_temporal = self.temporal
        if level is not None:
            try:
                new_level = self.level[level[0]-1:level[1]:1]
            ## may occur with a snippet where there is no level, but a range is
            ## requested.
            except TypeError:
                if self.level is None:
                    if list(level) == [1,1]:
                        new_level = self.level
                    else:
                        raise(ValueError('level subset requested but no levels available.'))
                else:
                    raise
        else:
            new_level = self.level
        if spatial_operation is not None and igeom is not None:
            if isinstance(igeom,Point):
                try:
                    _row = self.spatial.grid.row.value
                    _col = self.spatial.grid.column.value
                    row_idx = np.abs(_row - igeom.y)
                    row_idx = row_idx == row_idx.min()
                    col_idx = np.abs(_col - igeom.x)
                    col_idx = col_idx == col_idx.min()
                ## NcGridMatrixDimension correction
                except AttributeError:
                    _row = self.spatial.grid.row
                    _col = self.spatial.grid.column
                    coords = np.hstack([_col.reshape(-1,1),_row.reshape(-1,1)])
                    pt = np.array(igeom).reshape(1,2)
                    dist = np.empty(coords.shape[0],dtype=float)
                    norm = np.linalg.norm
                    for idx in range(coords.shape[0]):
                        dist[idx] = norm(coords[idx,:]-pt)
                    sel_idx = np.argmin(dist)
                    row_idx = _row == coords[sel_idx,1]
                    col_idx = _col == coords[sel_idx,0]
                new_grid = self.spatial.grid[row_idx,col_idx]
                new_vector = None
            else:
                if spatial_operation == 'intersects':
                    new_vector = self.spatial.vector.intersects(igeom)
                elif spatial_operation == 'clip':
                    new_vector = self.spatial.vector.clip(igeom)
                new_grid = new_vector.grid
            new_spatial = NcSpatialDimension(grid=new_grid,vector=new_vector,
                                             projection=self.spatial.projection,
                                             abstraction=self.spatial.abstraction)
        else:
            new_spatial = self.spatial
        ret = self.__class__(request_dataset=self.request_dataset,temporal=new_temporal,
         level=new_level,spatial=new_spatial,metadata=self.metadata,value=None)
        return(ret)
    
    def project(self,projection):
        raise(NotImplementedError)
        ## projection is only valid if the geometry has not been loaded. this is
        ## to limit the number of spatial operations. this is primary to ensure
        ## any masks created during a subset are not destroyed in the geometry
        ## resetting process.
        if self.spatial.vector._geom is not None:
            raise(NotImplementedError('project is only valid before geometries have been loaded.'))
        
        if self.spatial.grid.is_bounded:
            raise(NotImplementedError)
        
        ## project the rows and columns
        row = self.spatial.grid.row.value
        new_row = np.empty_like(row)
        col = self.spatial.grid.column.value
        new_col = np.empty_like(col)
        sr = self.spatial.projection.sr
        to_sr = projection.sr
        for row_idx in range(row.shape[0]):
            row_value = row[row_idx]
            for col_idx in range(col.shape[0]):
                col_value = col[col_idx]
                pt = Point(col_value,row_value)
                geom = CreateGeometryFromWkb(pt.wkb)
                geom.AssignSpatialReference(sr)
                geom.TransformTo(to_sr)
                new_col[col_idx] = geom.GetX()
            new_row[row_idx] = geom.GetY()
            
        ## update the rows and columns
        self.spatial.grid.row.value = new_row
        self.spatial.grid.column.value = new_col
        ## update the projection
        self.spatial.projection = projection
    
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
                for key,value in self.metadata['variables'].iteritems():
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
                for key2,value2 in self.metadata['variables'].iteritems():
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
            ## check for any initial slices
            slc = self._load_slice.get(kls.axis,slice(None))
            value = np.atleast_1d(ref['variable'][slc])
        except TypeError:
            raise(DummyDimensionEncountered(kls.axis))
        name = ref['variable']._name
        try:
            bounds = ref['bounds'][slc]
            name_bounds = ref['bounds']._name
        ## likely encountered empty bounds
        except TypeError:
            if ref['bounds'] is None:
                bounds = None
                name_bounds = None
            else:
                raise
        ret = kls(value=value,name=name,bounds=bounds,name_bounds=name_bounds)
        return(ret)
    
    @staticmethod
    def _sub_range_(arr):
        try:
            ret = (arr[0],arr[-1]+1)
        except IndexError:
            ret = (arr,arr+1)
        return(ret)
