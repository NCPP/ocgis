from collections import deque
import base
import numpy as np
from ocgis.interface.base.crs import CFWGS84, CoordinateReferenceSystem, WGS84, CFRotatedPole
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.util.helpers import iter_array, get_none_or_slice, \
    get_formatted_slice, get_reduced_slice, get_trimmed_array_by_mask,\
    get_added_slice
from shapely.geometry.point import Point
from ocgis import constants
import itertools
from shapely.geometry.polygon import Polygon
from copy import copy
from shapely.prepared import prep
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.exc import ImproperPolygonBoundsError, EmptySubsetError, SpatialWrappingError, MultipleElementsFound
from osgeo.ogr import CreateGeometryFromWkb
from shapely import wkb
import fiona
from shapely.geometry.geo import mapping, shape


class GeomMapping(object):
    """Used to simulate a dictionary key look up for data stored in 2-d ndarrays."""

    def __init__(self, uid, value):
        self.uid = uid
        self.value = value

    def __getitem__(self, key):
        sel = self.uid == key
        return self.value[sel][0]


class SingleElementRetriever(object):
    """
    Simplifies access to a spatial dimension with a single element.

    :param sdim:
    :type sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
    """

    def __init__(self, sdim):
        try:
            assert(sdim.shape == (1, 1))
        except AssertionError:
            raise MultipleElementsFound(sdim)
        self.sdim = sdim

    @property
    def crs(self):
        return self.sdim.crs

    @property
    def geom(self):
        return self.sdim.abstraction_geometry.value[0, 0]

    @property
    def properties(self):
        return self.sdim.properties

    @property
    def uid(self):
        return self.sdim.uid[0, 0]
                
                
class SpatialDimension(base.AbstractUidDimension):
    """
    :param grid: :class:`ocgis.interface.base.dimension.spatial.SpatialGridDimension`
    :param crs: :class:`ocgis.crs.CoordinateReferenceSystem`
    :param abstraction: str
    :param geom: :class:`ocgis.interface.base.dimension.spatial.SpatialGeometryDimension`
    """

    _ndims = 2
    _axis = 'SPATIAL'
    _attrs_slice = ('uid','grid','_geom')

    def __init__(self, *args, **kwds):
        self.grid = kwds.pop('grid', None)
        self.crs = kwds.pop('crs', None)
        self.abstraction = kwds.pop('abstraction', 'polygon')
        self._geom = kwds.pop('geom', None)

        # ## if a grid value is passed, then when it is reset
        # if self._grid is not None:
        # self._geom_to_grid = True
        # else:
        #     self._geom_to_grid = False

        # convert the input crs to CFWGS84 if they are equivalent
        if self.crs == CFWGS84():
            self.crs = CFWGS84()

        # remove row and col dimension keywords if they are present. we do not want to pass them to the superclass
        # constructor.
        row = kwds.pop('row', None)
        col = kwds.pop('col', None)

        ## attempt to build the geometry dimension
        point = kwds.pop('point', None)
        polygon = kwds.pop('polygon', None)
        geom_kwds = dict(point=point, polygon=polygon)
        if any([g != None for g in geom_kwds.values()]):
            self._geom = SpatialGeometryDimension(**geom_kwds)

        # attempt to construct some core dimensions if they are not passed at initialization
        if self._grid is None and self._geom is None:
            # try:
            self.grid = SpatialGridDimension(row=row, col=col)
            # except KeyError:
            # ocgis_lh(exc=ValueError('A SpatialDimension without "grid" or "geom" arguments requires a "row" and "column".'))

        super(SpatialDimension, self).__init__(*args, **kwds)

        assert self.abstraction in ('point', 'polygon', None)

    
    @property
    def abstraction_geometry(self):
        if self.abstraction is None:
            ret = self.geom.get_highest_order_abstraction()
        else:
            ret = getattr(self.geom, self.abstraction)
        return ret

    @property
    def geom(self):
        if self._geom is None:
            self._geom = SpatialGeometryDimension(grid=self.grid, uid=self.grid.uid)
        return self._geom
    
    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        if value is not None:
            assert(isinstance(value, SpatialGridDimension))
        self._grid = value

    @property
    def is_unwrapped(self):
        """
        Return ``True`` if the coordinates of the spatial data have a 0 to 360 longitudinal domain. Returns ``False``
        for non-WGS84 coordinate systems.
        """

        try:
            ret = self.crs.get_is_360(self)
        # None and non-WGS84 coordinate systems have no determination for wrapping
        except AttributeError:
            ret = False
        return ret

    @property
    def shape(self):
        if self.grid is None:
            ret = self.geom.shape
        else:
            ret = self.grid.shape
        return ret

    @property
    def single(self):
        return SingleElementRetriever(self)
        
    @property
    def weights(self):
        if self.geom is None:
            ret = self.grid.weights
        else:
            try:
                ret = self.geom.polygon.weights
            except ImproperPolygonBoundsError:
                ret = self.geom.point.weights
        return ret

    def assert_uniform_mask(self):
        """
        Check that the mask for the major spatial components are equivalent. This will only test loaded elements.

        :raises: AssertionError
        """

        to_compare = []
        if self._grid is not None:
            to_compare.append(self._grid.value[0].mask)
            to_compare.append(self._grid.value[1].mask)
        if self._geom is not None:
            if self._geom._point is not None:
                to_compare.append(self._geom._point.value.mask)
            if self._geom._polygon is not None:
                to_compare.append(self._geom._polygon.value.mask)
        to_compare.append(self.uid.mask)

        for arr1, arr2 in itertools.combinations(to_compare, 2):
            assert np.all(arr1 == arr2)

    @classmethod
    def from_records(cls, records, crs=None):
        """
        Create a :class:`ocgis.interface.base.dimension.SpatialDimension` from Fiona-like records.

        :param records: A sequence of records returned from an Fiona file object.
        :type records: sequence
        :param crs: If ``None``, default to :attr:`~ocgis.constants.default_coordinate_system`.
        :type crs: dict or :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :rtype: :class:`ocgis.interface.base.dimension.SpatialDimension`
        """

        if not isinstance(crs, CoordinateReferenceSystem):
            # if there is no crs dictionary passed, assume WGS84
            crs = crs or constants.default_coordinate_system.value
            crs = CoordinateReferenceSystem(crs=crs)

        # these are mappings used to construct the SpatialDimension
        mapping_geometry = {SpatialGeometryPolygonDimension: ('Polygon', 'MultiPolygon'),
                            SpatialGeometryPointDimension: ('Point', 'MultiPoint')}
        mapping_kwds = {SpatialGeometryPolygonDimension: 'polygon',
                        SpatialGeometryPointDimension: 'point'}

        # holds data types for the property structure array
        dtype = []
        # holds geometry objects
        deque_geoms = deque()
        # holds unique identifiers
        deque_uid = deque()

        build = True
        for ctr, record in enumerate(records, start=1):

            # get the geometry from a keyword present on the input dictionary or construct from the coordinates sequence
            try:
                current_geom = record['geom']
            except KeyError:
                current_geom = shape(record['geometry'])
            deque_geoms.append(current_geom)

            # this is to set up the properties array
            if build:
                build = False

                if 'UGID' in record['properties']:
                    has_ugid = True
                else:
                    has_ugid = False

                for k, v in record['properties'].iteritems():
                    the_type = type(v)
                    if the_type == unicode:
                        the_type = object
                    if isinstance(v, basestring):
                        the_type = object
                    dtype.append((str(k), the_type))
                properties = np.empty(0, dtype=dtype)
                property_order = record['properties'].keys()

            # the UGID may be present as a property. otherwise the enumeration counter is used for the identifier.
            if has_ugid:
                to_append = int(record['properties']['UGID'])
            else:
                to_append = ctr
            deque_uid.append(to_append)

            # append to the properties array
            properties_new = np.empty(1, dtype=dtype)
            properties_new[0] = tuple([record['properties'][key] for key in property_order])
            properties = np.append(properties, properties_new)

        # fill the geometry array. to avoid having the geometry objects turned into coordinates, fill by index...
        geoms = np.empty((1, len(deque_geoms)), dtype=object)
        for idx in range(geoms.shape[1]):
            geoms[0, idx] = deque_geoms[idx]

        # convert the unique identifiers to an array
        uid = np.array(deque_uid).reshape(*geoms.shape)

        # this will choose the appropriate geometry dimension
        geom_type = geoms[0, 0].geom_type
        for k, v in mapping_geometry.iteritems():
            if geom_type in v:
                klass = k
                break

        # this constructs the geometry dimension
        dim_geom_type = klass(value=geoms)
        # arguments to geometry dimension
        kwds = {mapping_kwds[klass]: dim_geom_type}
        dim_geom = SpatialGeometryDimension(**kwds)

        sdim = SpatialDimension(geom=dim_geom, uid=uid, properties=properties, crs=crs, abstraction=mapping_kwds[klass])

        return sdim
    
    def get_clip(self, polygon, return_indices=False, use_spatial_index=True, select_nearest=False):
        assert(type(polygon) in (Polygon, MultiPolygon))
        
        ret, slc = self.get_intersects(polygon, return_indices=True, use_spatial_index=use_spatial_index, select_nearest=select_nearest)
        
        ## clipping with points is okay...
        try:
            ref_value = ret.geom.polygon.value
        except ImproperPolygonBoundsError:
            ref_value = ret.geom.point.value
        for (row_idx, col_idx), geom in iter_array(ref_value, return_value=True):
            ref_value[row_idx, col_idx] = geom.intersection(polygon)
                
        if return_indices:
            ret = (ret, slc)
        
        return(ret)
        
    def get_intersects(self, polygon, return_indices=False, use_spatial_index=True, select_nearest=False):
        ret = copy(self)

        ## based on the set spatial abstraction, decide if bounds should be used
        ## for subsetting the row and column dimensions
        use_bounds = False if self.abstraction == 'point' else True

        if type(polygon) in (Point,MultiPoint):
            raise(ValueError('Only Polygons and MultiPolygons are acceptable geometry types for intersects operations.'))
        elif type(polygon) in (Polygon,MultiPolygon):
            ## for a polygon subset, first the grid is subsetted by the bounds
            ## of the polygon object. the intersects operations is then performed
            ## on the polygon/point representation as appropriate.
            minx,miny,maxx,maxy = polygon.bounds
            if self.grid is None:
                raise(NotImplementedError)
            else:
                ## reset the geometries
                ret._geom = None
                ## subset the grid by its bounding box
                ret.grid,slc = self.grid.get_subset_bbox(minx,miny,maxx,maxy,
                                                         return_indices=True,
                                                         use_bounds=use_bounds)
                ## update the unique identifier to copy the grid uid
                ret.uid = ret.grid.uid
                ## attempt to mask the polygons
                try:
                    ## only use the polygons if the abstraction indicates as much
                    ret._geom._polygon = ret.geom.polygon.get_intersects_masked(polygon,
                     use_spatial_index=use_spatial_index)
                    grid_mask = ret.geom.polygon.value.mask
                except ImproperPolygonBoundsError:
                    ret._geom._point = ret.geom.point.get_intersects_masked(polygon,
                     use_spatial_index=use_spatial_index)
                    grid_mask = ret.geom.point.value.mask
                ## transfer the geometry mask to the grid mask
                ret.grid.value.mask[:,:,:] = grid_mask.copy()
        else:
            raise(NotImplementedError)

        ## barbed and circular geometries may result in rows and or columns being
        ## entirely masked. these rows and columns should be trimmed.
        _,adjust = get_trimmed_array_by_mask(ret.get_mask(), return_adjustments=True)
        ## use the adjustments to trim the returned data object
        ret = ret[adjust['row'], adjust['col']]

        if select_nearest:
            try:
                if self.abstraction == 'point':
                    raise(ImproperPolygonBoundsError)
                else:
                    target_geom = ret.geom.polygon.value
            except ImproperPolygonBoundsError:
                target_geom = ret.geom.point.value
            distances = {}
            centroid = polygon.centroid
            for select_nearest_index, geom in iter_array(target_geom, return_value=True):
                distances[centroid.distance(geom)] = select_nearest_index
            select_nearest_index = distances[min(distances.keys())]
            ret = ret[select_nearest_index[0], select_nearest_index[1]]

        if return_indices:
            ## adjust the returned slices if necessary
            if select_nearest:
                ret_slc = select_nearest_index
            else:
                ret_slc = [None, None]
                ret_slc[0] = get_added_slice(slc[0], adjust['row'])
                ret_slc[1] = get_added_slice(slc[1], adjust['col'])
            ret = (ret, tuple(ret_slc))

        return(ret)
    
    def get_geom_iter(self,target=None,as_multipolygon=True):
        target = target or self.abstraction
        if target is None:
            value = self.geom.get_highest_order_abstraction().value
        else:
            value = getattr(self.geom,target).value

        ## no need to attempt and convert to MultiPolygon if we are working with
        ## point data.
        if as_multipolygon and target == 'point':
            as_multipolygon = False

        r_uid = self.uid
        for (row_idx,col_idx),geom in iter_array(value,return_value=True):
            if as_multipolygon:
                if isinstance(geom,Polygon):
                    geom = MultiPolygon([geom])
            uid = r_uid[row_idx,col_idx]
            yield(row_idx,col_idx,geom,uid)

    def get_mask(self):
        if self.grid is None:
            if self.geom.point is None:
                ret = self.geom.polygon.value.mask
            else:
                ret = self.geom.point.value.mask
        else:
            ret = self.grid.value.mask[0,:,:]
        return(ret.copy())

    def set_grid_bounds_from_geometry(self):
        """
        Sets the grid bounds to a masked array with dimensions (nrows, ncols, 4). The third dimension indices correspond
        to:
         0 - min column
         1 - min row
         2 - max column
         3 - max row
        """

        if self.grid is None:
            ocgis_lh(exc=ValueError('Grid bounds may only be computed when a grid is present.'))

        if self.geom.polygon is not None:
            shp = list(self.geom.polygon.shape) + [4]
            fill = np.empty(shp)
            fill_mask = np.zeros(fill.shape, dtype=bool)
            r_mask = self.geom.polygon.value.mask
            for (idx_row, idx_col), geom in iter_array(self.geom.polygon.value, use_mask=False, return_value=True):
                fill[idx_row, idx_col, :] = geom.bounds
                fill_mask[idx_row, idx_col, :] = r_mask[idx_row, idx_col]
            fill = np.ma.array(fill, mask=fill_mask)
        else:
            raise NotImplementedError
        self.grid.bounds = fill

    def set_grid_value_from_geometry(self):
        """
        Set the grid attribute to a :class:`ocgis.interface.base.dimension.spatial.SpatialGridDimension` object created
        from point geometry centroids.
        """
        r_point = self.geom.point.value
        fill = np.zeros([2]+list(r_point.shape))
        for (row, col), point in iter_array(r_point, return_value=True, use_mask=False):
            fill[0, row, col] = point.y
            fill[1, row, col] = point.x
        r_mask = self.get_mask()
        fill_mask = np.zeros(fill.shape, dtype=bool)
        fill_mask[0, ...] = r_mask
        fill_mask[1, ...] = r_mask
        fill = np.ma.array(fill, mask=fill_mask)
        self.grid = SpatialGridDimension(value=fill)

    def unwrap(self):
        try:
            self.crs.unwrap(self)
        except AttributeError:
            if self.crs is None or self.crs != WGS84():
                msg = 'Only WGS84 coordinate systems may be unwrapped.'
                raise(SpatialWrappingError(msg))

    def update_crs(self, to_crs):
        """
        :param to_crs: The destination CRS object.
        :type to_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        """

        try:
            # if the crs values are the same, pass through
            if to_crs != self.crs:
                to_sr = to_crs.sr
                from_sr = self.crs.sr

                if self.geom.point is not None:
                    self.geom.point.update_crs(to_sr, from_sr)
                try:
                    self.geom.polygon.update_crs(to_sr, from_sr)
                except ImproperPolygonBoundsError:
                    pass

                if self.grid is not None and self.geom.point is not None:
                    r_grid_value = self.grid.value.data
                    r_point_value = self.geom.point.value.data
                    for (idx_row, idx_col), geom in iter_array(r_point_value, return_value=True, use_mask=False):
                        x, y = geom.x, geom.y
                        r_grid_value[0, idx_row, idx_col] = y
                        r_grid_value[1, idx_row, idx_col] = x
                    ## remove row and columns if they exist as this requires interpolation
                    ## to make them vectors again.
                    self.grid.row = None
                    self.grid.col = None
                ## if there is not point dimension, then a grid representation is not
                ## possible. mask the grid values accordingly.
                elif self.grid is not None and self.geom.point is None:
                    self.grid.value.mask = True

                self.crs = to_crs

        # likely a rotated pole coordinate system.
        except RuntimeError:
            try:
                _crs = self.crs
                """:type: ocgis.interface.base.crs.CFRotatedPole"""
                new_spatial = _crs.get_rotated_pole_transformation(self)
            # likely an inverse transformation if the destination crs is rotated pole.
            except AttributeError:
                new_spatial = to_crs.get_rotated_pole_transformation(self, inverse=True)
            self.__dict__ = new_spatial.__dict__
            self.crs = to_crs

    def wrap(self):
        try:
            self.crs.wrap(self)
        except AttributeError:
            if self.crs is None or self.crs != WGS84():
                msg = 'Only WGS84 coordinate systems may be wrapped.'
                raise(SpatialWrappingError(msg))

    def write_fiona(self,path,target='polygon',driver='ESRI Shapefile'):
        attr = getattr(self.geom,target)
        attr.write_fiona(path,self.crs.value,driver=driver)
        return(path)
    
    def _format_uid_(self,value):
        return(np.atleast_2d(value))
    
    def _get_sliced_properties_(self,slc):
        if self.properties is not None:
            ## determine major axis
            major = self.shape.index(max(self.shape))
            return(self.properties[slc[major]])
        else:
            return(None)
        
    def _get_uid_(self):
        if self._geom is not None:
            ret = self._geom.uid
        else:
            ret = self.grid.uid
        return(ret)

    
class SpatialGridDimension(base.AbstractUidValueDimension):
    _axis = 'GRID'
    _ndims = 2
    _attrs_slice = None

    def __init__(self, *args, **kwargs):
        self.row = kwargs.pop('row', None)
        self.col = kwargs.pop('col', None)
        self.bounds = kwargs.pop('bounds', None)
        self._row_src_idx = kwargs.pop('row_src_idx', None)
        self._col_src_idx = kwargs.pop('col_src_idx', None)

        super(SpatialGridDimension, self).__init__(*args, **kwargs)

        if self._value is None:
            if self.row is None or self.col is None:
                msg = 'Without a value, a row and column dimension are required.'
                raise(ValueError(msg))
        
    def __getitem__(self,slc):
        slc = get_formatted_slice(slc,2)
        
        uid = self.uid[slc]
        
        if self._value is not None:
            value = self._value[:,slc[0],slc[1]]
        else:
            value = None
        
        if self.row is not None:
            row = self.row[slc[0]]
            col = self.col[slc[1]]
        else:
            row = None
            col = None
        
        ret = copy(self)

        if ret.bounds is not None:
            ret.bounds = ret.bounds[slc[0], slc[1], :]
        
        if self._row_src_idx is not None:
            ret._row_src_idx = self._row_src_idx[slc[0]]
            ret._col_src_idx = self._col_src_idx[slc[1]]
        
        ret.uid = uid
        ret._value = value
        ret.row = row
        ret.col = col
            
        return(ret)
    
    @property
    def extent(self):
        if self.row is None:
            minx = self.value[1,:,:].min()
            miny = self.value[0,:,:].min()
            maxx = self.value[1,:,:].max()
            maxy = self.value[0,:,:].max()
        else:
            if self.row.bounds is None:
                minx = self.col.value.min()
                miny = self.row.value.min()
                maxx = self.col.value.max()
                maxy = self.row.value.max()
            else:
                minx = self.col.bounds.min()
                miny = self.row.bounds.min()
                maxx = self.col.bounds.max()
                maxy = self.row.bounds.max()
        return(minx,miny,maxx,maxy)
        
    @property
    def resolution(self):
        try:
            ret = np.mean([self.row.resolution,self.col.resolution])
        except AttributeError:
            resolution_limit = int(constants.resolution_limit)/2
            r_value = self.value[:,0:resolution_limit,0:resolution_limit]
            rows = np.mean(np.diff(r_value[0,:,:],axis=0))
            cols = np.mean(np.diff(r_value[1,:,:],axis=1))
            ret = np.mean([rows,cols])
        return(ret)
    
    @property
    def shape(self):
        try:
            ret = (len(self.row), len(self.col))
        # occurs if either of these are empty. get the shape from the grid value.
        except TypeError:
            ret = (self.value.shape[1], self.value.shape[2])
        return ret
        
    def get_subset_bbox(self,min_col,min_row,max_col,max_row,return_indices=False,closed=True,
                        use_bounds=True):
        assert(min_row <= max_row)
        assert(min_col <= max_col)
        
        if self.row is None:
            r_row = self.value[0,:,:]
            real_idx_row = np.arange(0,r_row.shape[0])
            r_col = self.value[1,:,:]
            real_idx_col = np.arange(0,r_col.shape[1])
            
            if closed:
                lower_row = r_row > min_row
                upper_row = r_row < max_row
                lower_col = r_col > min_col
                upper_col = r_col < max_col
            else:
                lower_row = r_row >= min_row
                upper_row = r_row <= max_row
                lower_col = r_col >= min_col
                upper_col = r_col <= max_col
            
            idx_row = np.logical_and(lower_row,upper_row)
            idx_col = np.logical_and(lower_col,upper_col)
            
            keep_row = np.any(idx_row,axis=1)
            keep_col = np.any(idx_col,axis=0)
            
            ## slice reduction may fail due to empty bounding box returns. catch
            ## these value errors and repurpose as subset errors.
            try:
                row_slc = get_reduced_slice(real_idx_row[keep_row])
            except ValueError:
                if real_idx_row[keep_row].shape[0] == 0:
                    raise(EmptySubsetError(origin='Y'))
                else:
                    raise
            try:
                col_slc = get_reduced_slice(real_idx_col[keep_col])
            except ValueError:
                if real_idx_col[keep_col].shape[0] == 0:
                    raise(EmptySubsetError(origin='X'))
                else:
                    raise
            
            new_mask = np.invert(np.logical_or(idx_row,idx_col)[row_slc,col_slc])
            
        else:
            new_row,row_indices = self.row.get_between(min_row,max_row,return_indices=True,closed=closed,use_bounds=use_bounds)
            new_col,col_indices = self.col.get_between(min_col,max_col,return_indices=True,closed=closed,use_bounds=use_bounds)
            row_slc = get_reduced_slice(row_indices)
            col_slc = get_reduced_slice(col_indices)
        
        ret = self[row_slc,col_slc]
        
        try:
            grid_mask = np.zeros((2,new_mask.shape[0],new_mask.shape[1]),dtype=bool)
            grid_mask[:,:,:] = new_mask
            ret._value = np.ma.array(ret._value,mask=grid_mask)
            ret.uid = np.ma.array(ret.uid,mask=new_mask)
        except UnboundLocalError:
            if self.row is not None:
                pass
            else:
                raise

        if return_indices:
            ret = (ret,(row_slc,col_slc))

        return(ret)
    
    def _format_private_value_(self,value):
        if value is None:
            ret = None
        else:
            assert(len(value.shape) == 3)
            assert(value.shape[0] == 2)
            assert(isinstance(value,np.ma.MaskedArray))
            ret = value
        return(ret)
        
    def _get_slice_(self,state,slc):

        if self._value is None:
            state._value = None
        else:
            state._value = state.value[:,slc[0],slc[1]]
        if state.row is not None:
            state.row = state.row[slc[0]]
            state.col = state.col[slc[1]]
        
        return(state)
        
    def _get_uid_(self):
        if self._value is None:
            shp = len(self.row),len(self.col)
        else:
            shp = self._value.shape[1],self._value.shape[2]
        ret = np.arange(1,(shp[0]*shp[1])+1,dtype=constants.np_int).reshape(shp)
        ret = np.ma.array(ret,mask=False)
        return(ret)
    
    def _get_value_(self):
        ## assert types of row and column are equivalent
        if self.row.value.dtype != self.col.value.dtype:
            ocgis_lh(exc=ValueError('Row and column data types differ! They must be equivalent.'))
        ## fill the centroids
        fill = np.empty((2,self.row.shape[0],self.col.shape[0]),dtype=self.row.value.dtype)
        fill = np.ma.array(fill,mask=False)
        col_coords,row_coords = np.meshgrid(self.col.value,self.row.value)
        fill[0,:,:] = row_coords
        fill[1,:,:] = col_coords
        return(fill)
    
    
class SpatialGeometryDimension(base.AbstractUidDimension):
    _axis = 'GEOM'
    _ndims = 2
    _attrs_slice = ('uid','grid','_point','_polygon')
    
    def __init__(self,*args,**kwds):
        self.grid = kwds.pop('grid',None)
        self._point = kwds.pop('point',None)
        self._polygon = kwds.pop('polygon',None)
        
        super(SpatialGeometryDimension,self).__init__(*args,**kwds)

        if self.grid is None and self._point is None and self._polygon is None:
            msg = 'At minimum, a grid, point, or polygon dimension is required.'
            raise(ValueError(msg))
    
    @property
    def point(self):
        if self._point is None and self.grid is not None:
            self._point = SpatialGeometryPointDimension(grid=self.grid,uid=self.grid.uid)
        return(self._point)
    
    @property
    def polygon(self):
        if self._polygon is None:
            if self.grid is None:
                msg = 'Constructing a polygon dimension requires a grid dimension.'
                raise ImproperPolygonBoundsError(msg)
            else:
                self._polygon = SpatialGeometryPolygonDimension(grid=self.grid,uid=self.grid.uid)
        return(self._polygon)
    
    @property
    def shape(self):
        if self.point is None:
            ret = self.polygon.shape
        else:
            ret = self.point.shape
        return(ret)
    
    def get_highest_order_abstraction(self):
        """
        Return the highest order abstraction geometry with preference given by:
         1. Polygon
         2. Point
        """

        try:
            ret = self.polygon
            # if the polygon is a NoneType, return the point dimension
            if ret is None:
                ret = self.point
        except ImproperPolygonBoundsError:
            ret = self.point
        return ret
    
    def get_iter(self):
        raise(NotImplementedError)
        
    def _get_slice_(self,state,slc):
        state._point = get_none_or_slice(state._point,slc)
        state._polygon = get_none_or_slice(state._polygon,slc)
        return(state)
        
    def _get_uid_(self):
        if self._point is not None:
            ret = self._point.uid
        elif self._polygon is not None:
            ret = self._polygon.uid
        else:
            ret = self.grid.uid
        return(ret)


class SpatialGeometryPointDimension(base.AbstractUidValueDimension):
    _axis = 'POINT'
    _ndims = 2
    _attrs_slice = ('uid','_value','grid')
    _geom_type = 'Point'
    
    def __init__(self,*args,**kwds):
        self.grid = kwds.pop('grid',None)
        
        super(SpatialGeometryPointDimension,self).__init__(*args,**kwds)
        
    @property
    def weights(self):
        ret = np.ones(self.value.shape,dtype=constants.np_float)
        ret = np.ma.array(ret,mask=self.value.mask)
        return(ret)
        
    def get_intersects_masked(self,polygon,use_spatial_index=True):
        '''
        :param polygon: The Shapely geometry to use for subsetting.
        :type polygon: :class:`shapely.geometry.Polygon' or :class:`shapely.geometry.MultiPolygon'
        :param bool use_spatial_index: If ``False``, do not use the :class:`rtree.index.Index`
         for spatial subsetting. If the geometric case is simple, it may marginally
         improve execution times to turn this off. However, turning this off for 
         a complex case will negatively impact (significantly) spatial operation
         execution times.
        :raises: NotImplementedError, EmptySubsetError
        :returns: :class:`ocgis.interface.base.dimension.spatial.SpatialGeometryPointDimension`
        '''
        
        ## only polygons are acceptable for subsetting. if a point is required,
        ## buffer it.
        if type(polygon) not in (Polygon,MultiPolygon):
            raise(NotImplementedError(type(polygon)))
        
        ## return a shallow copy of self
        ret = copy(self)
        ## create the fill array and reference the mask. this is the outpout
        ## geometry value array.
        fill = np.ma.array(ret.value,mask=True)
        ref_fill_mask = fill.mask
        
        ## this is the path if a spatial index is used.
        if use_spatial_index:
            ## keep this as a local import as it is not a required dependency
            from ocgis.util.spatial.index import SpatialIndex
            ## create the index object and reference import members
            si = SpatialIndex()
            _add = si.add
            _value = self.value
            ## add the geometries to the index
            for (ii,jj),id_value in iter_array(self.uid,return_value=True):
                _add(id_value,_value[ii,jj])
            ## this mapping simulates a dictionary for the item look-ups from
            ## two-dimensional arrays
            geom_mapping = GeomMapping(self.uid,self.value)
            _uid = ret.uid
            ## return the identifiers of the objects intersecting the target geometry
            ## and update the mask accordingly
            for intersect_id in si.iter_intersects(polygon,geom_mapping,keep_touches=False):
                sel = _uid == intersect_id
                ref_fill_mask[sel] = False
        ## this is the slower simpler case
        else:
            ## prepare the polygon for faster spatial operations
            prepared = prep(polygon)
            ## we are not keeping touches at this point. remember the mask is an
            ## inverse.
            for (ii,jj),geom in iter_array(self.value,return_value=True):
                bool_value = False
                if prepared.intersects(geom):
                    if polygon.touches(geom):
                        bool_value = True
                else:
                    bool_value = True
                ref_fill_mask[ii,jj] = bool_value
        
        ## if everything is masked, this is an empty subset
        if ref_fill_mask.all():
            raise(EmptySubsetError(self.name))
        
        ## set the returned value to the fill array
        ret._value = fill
        ## also update the unique identifier array
        ret.uid = np.ma.array(ret.uid,mask=fill.mask.copy())
        
        return(ret)
    
    def update_crs(self,to_sr,from_sr):
        ## we are modifying the original source data and need to copy the new
        ## values.
        new_value = self.value.copy()
        ## be sure and project masked geometries to maintain underlying geometries
        ## for masked values.
        r_value = new_value.data
        r_loads = wkb.loads
        for (idx_row,idx_col),geom in iter_array(r_value,return_value=True,use_mask=False):
            ogr_geom = CreateGeometryFromWkb(geom.wkb)
            ogr_geom.AssignSpatialReference(from_sr)
            ogr_geom.TransformTo(to_sr)
            r_value[idx_row,idx_col] = r_loads(ogr_geom.ExportToWkb())
        self._value = new_value
            
    def write_fiona(self,path,crs,driver='ESRI Shapefile'):
        schema = {'geometry':self._geom_type,
                  'properties':{'UGID':'int'}}
        ref_prep = self._write_fiona_prep_geom_
        ref_uid = self.uid
        
        with fiona.open(path,'w',driver=driver,crs=crs,schema=schema) as f:
            for (ii,jj),geom in iter_array(self.value,return_value=True):
                geom = ref_prep(geom)
                uid = int(ref_uid[ii,jj])
                feature = {'properties':{'UGID':uid},'geometry':mapping(geom)}
                f.write(feature)
                feature = {'UGID'}
        
        return(path)
        
    def _write_fiona_prep_geom_(self,geom):
        return(geom)
        
    def _format_private_value_(self,value):
        if value is not None:
            try:
                assert(len(value.shape) == 2)
                ret = value
            except (AssertionError,AttributeError):
                ocgis_lh(exc=ValueError('Geometry values must come in as 2-d NumPy arrays to avoid array interface modifications by shapely.'))
        else:
            ret = None
        ret = self._get_none_or_array_(ret,masked=True)
        return(ret)
    
    def _get_geometry_fill_(self,shape=None):
        if shape is None:
            shape = (self.grid.shape[0],self.grid.shape[1])
            mask = self.grid.value[0].mask
        else:
            mask = False
        fill = np.ma.array(np.zeros(shape),mask=mask,dtype=object)

        return(fill)
    
    def _get_value_(self):
        # we are interested in creating geometries for all the underly coordinates regardless if the data is masked
        ref_grid = self.grid.value.data
        
        fill = self._get_geometry_fill_()
        r_data = fill.data
        for idx_row,idx_col in iter_array(ref_grid[0],use_mask=False):
            y = ref_grid[0,idx_row,idx_col]
            x = ref_grid[1,idx_row,idx_col]
            pt = Point(x,y)
            r_data[idx_row,idx_col] = pt
        return(fill)
    
    
class SpatialGeometryPolygonDimension(SpatialGeometryPointDimension):
    _geom_type = 'MultiPolygon'
    _axis = 'POLYGON'
    
    def __init__(self,*args,**kwds):
        super(SpatialGeometryPolygonDimension,self).__init__(*args,**kwds)
        
        if self._value is None:
            # we can construct from a grid dimension having bounds
            if self.grid is None:
                msg = 'A grid dimension is required for constructing a polygon dimension without a value.'
                raise(ImproperPolygonBoundsError(msg))
            else:
                if self.grid.bounds is None:
                    if self.grid.row is None or self.grid.col is None:
                        raise(ImproperPolygonBoundsError('Polygon dimensions require a grid dimension with row and column dimensions with bounds.'))
                    else:
                        if self.grid.row.bounds is None or self.grid.col.bounds is None:
                            raise(ImproperPolygonBoundsError('Polygon dimensions require row and column dimension bounds to have delta > 0.'))

    @property
    def area(self):
        r_value = self.value
        fill = np.ones(r_value.shape,dtype=constants.np_float)
        fill = np.ma.array(fill,mask=r_value.mask)
        for (ii,jj),geom in iter_array(r_value,return_value=True):
            fill[ii,jj] = geom.area
        return(fill)
    
    @property
    def weights(self):
        return(self.area/self.area.max())

    def _get_value_(self):
        fill = self._get_geometry_fill_()
        r_data = fill.data
        try:
            ref_row_bounds = self.grid.row.bounds
            ref_col_bounds = self.grid.col.bounds
            for idx_row, idx_col in itertools.product(range(ref_row_bounds.shape[0]), range(ref_col_bounds.shape[0])):
                row_min, row_max = ref_row_bounds[idx_row, :].min(), ref_row_bounds[idx_row, :].max()
                col_min, col_max = ref_col_bounds[idx_col, :].min(), ref_col_bounds[idx_col, :].max()
                r_data[idx_row, idx_col] = Polygon(
                    [(col_min, row_min), (col_min, row_max), (col_max, row_max), (col_max, row_min)])
        # the grid dimension may not have row/col or row/col bounds
        except AttributeError:
            # we want geometries for everything even if masked
            r_bounds = self.grid.bounds.data
            range_row = range(self.grid.shape[0])
            range_col = range(self.grid.shape[1])
            for row, col in itertools.product(range_row, range_col):
                col_min, row_min, col_max, row_max = r_bounds[row, col, :]
                polygon = Polygon([(col_min, row_min), (col_min, row_max), (col_max, row_max), (col_max, row_min)])
                r_data[row, col] = polygon
        return fill
