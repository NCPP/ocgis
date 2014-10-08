from copy import copy, deepcopy
import tempfile
import itertools
from osgeo.osr import SpatialReference
from fiona.crs import from_string, to_string
import numpy as np
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.exc import SpatialWrappingError, ProjectionCoordinateNotFound,\
    ProjectionDoesNotMatch, ImproperPolygonBoundsError, CornersUnavailable
from ocgis.util.spatial.wrap import Wrapper
from ocgis.util.helpers import iter_array
from shapely.geometry.multipolygon import MultiPolygon
import abc
import logging
from shapely.geometry.multipoint import MultiPoint


class CoordinateReferenceSystem(object):
    
    def __init__(self, value=None, proj4=None, epsg=None):
        if value is None:
            if proj4 is not None:
                value = from_string(proj4)
            elif epsg is not None:
                sr = SpatialReference()
                sr.ImportFromEPSG(epsg)
                value = from_string(sr.ExportToProj4())
            else:
                msg = 'A value dictionary, PROJ.4 string, or EPSG code is required.'
                raise ValueError(msg)
        else:
            # remove unicode to avoid strange issues with proj and and fiona.
            for k, v in value.iteritems():
                if type(v) == unicode:
                    value[k] = str(v)
                else:
                    try:
                        value[k] = v.tolist()
                    # this may be a numpy arr that needs conversion
                    except AttributeError:
                        continue
            
        sr = SpatialReference()
        sr.ImportFromProj4(to_string(value))
        self.value = from_string(sr.ExportToProj4())
    
        try:
            assert self.value != {}
        except AssertionError:
            msg = 'Empty CRS: The conversion to PROJ.4 may have failed. The CRS value is: {0}'.format(value)
            ocgis_lh(logger='crs', exc=ValueError(msg))
    
    def __eq__(self, other):
        try:
            if self.sr.IsSame(other.sr) == 1:
                ret = True
            else:
                ret = False
        except AttributeError:
            # likely a nonetype of other object type
            if other is None or not isinstance(other, self.__class__):
                ret = False
            else:
                raise
        return ret
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.value)

    @property
    def proj4(self):
        return self.sr.ExportToProj4()
    
    @property
    def sr(self):
        sr = SpatialReference()
        sr.ImportFromProj4(to_string(self.value))
        return sr


class WrappableCoordinateReferenceSystem(object):
    """Mean to be used in mixin class that can be wrapped."""

    @classmethod
    def get_is_360(cls, spatial):
        """
        :param spatial:
        :type spatial: :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        if not isinstance(spatial.crs, WrappableCoordinateReferenceSystem):
            msg = 'Wrapped state may only be determined for geographic (i.e. spherical) coordinate systems.'
            raise SpatialWrappingError(msg)

        try:
            if spatial.grid.col.bounds is None:
                check = spatial.grid.col.value
            else:
                check = spatial.grid.col.bounds
        except AttributeError as e:
            # column dimension is likely missing
            try:
                if spatial.grid.col is None:
                    try:
                        check = spatial.grid.corners[1]
                    except CornersUnavailable:
                        check = spatial.grid.value[1, :, :]
                else:
                    ocgis_lh(exc=e)
            except AttributeError:
                # there may be no grid, access the geometries directly
                try:
                    geoms_to_check = spatial.geom.polygon.value
                except ImproperPolygonBoundsError:
                    geoms_to_check = spatial.geom.point.value
                geoms_to_check = geoms_to_check.compressed()

                # if this is switched to true, there are geometries with coordinate values less than 0
                for geom in geoms_to_check:
                    if type(geom) in [MultiPolygon, MultiPoint]:
                        it = geom
                    else:
                        it = [geom]
                    for sub_geom in it:
                        try:
                            coords = np.array(sub_geom.exterior.coords)
                            if np.any(coords > 180.):
                                return True
                        ## might be checking a point
                        except AttributeError:
                            coords = np.array(sub_geom)
                            if np.any(coords > 180.):
                                return True
                return False

        if np.any(check > 180.):
            ret = True
        else:
            ret = False

        return ret

    def unwrap(self, spatial):
        """
        :type spatial: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        if not self.get_is_360(spatial):
            # unwrap the geometries
            unwrap = Wrapper().unwrap
            to_wrap = self._get_to_wrap_(spatial)
            for tw in to_wrap:
                if tw is not None:
                    geom = tw.value.data
                    for (ii,jj),to_wrap in iter_array(geom,return_value=True,use_mask=False):
                        geom[ii,jj] = unwrap(to_wrap)
            if spatial._grid is not None:
                ref = spatial.grid.value.data[1,:,:]
                select = ref < 0
                ref[select] += 360
                if spatial.grid.col is not None:
                    ref = spatial.grid.col.value
                    select = ref < 0
                    ref[select] += 360
                    if spatial.grid.col.bounds is not None:
                        ref = spatial.grid.col.bounds
                        select = ref < 0
                        ref[select] += 360

                # attempt to to unwrap the grid corners if they exist
                try:
                    select = spatial.grid.corners[1] < 0
                    spatial.grid.corners[1][select] += 360
                except CornersUnavailable:
                    pass

        else:
            ocgis_lh(exc=SpatialWrappingError('Data already has a 0 to 360 coordinate system.'))

    def wrap(self,spatial):
        """
        Wrap ``spatial`` properties using a 180 degree prime meridian. If bounds _contain_ the prime meridian, the
        object may not be appropriately wrapped and bounds are removed.

        :param spatial: The object to wrap inplace.
        :type spatial: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        if self.get_is_360(spatial):
            # wrap the geometries if they are available
            wrap = Wrapper().wrap
            to_wrap = self._get_to_wrap_(spatial)
            for tw in to_wrap:
                if tw is not None:
                    geom = tw.value.data
                    for (ii,jj),to_wrap in iter_array(geom,return_value=True,use_mask=False):
                        geom[ii,jj] = wrap(to_wrap)

            # if there is a grid present, wrap its associated elements
            if spatial.grid is not None:
                bounds_cross_meridian = False
                ref = spatial.grid.value.data[1, :, :]
                select = ref > 180
                ref[select] -= 360
                if spatial.grid.col is not None:
                    ref = spatial.grid.col.value
                    select = ref > 180
                    ref[select] -= 360
                    if spatial.grid.col.bounds is not None:
                        ref = spatial.grid.col.bounds
                        # check for bounds containing the prime meridian. to avoid adjusting data, bounds will need to
                        # be removed.
                        bounds_min = np.min(ref, axis=1)
                        bounds_max = np.max(ref, axis=1)
                        select_min = bounds_min <= 180
                        select_max = bounds_max > 180
                        select_cross = np.logical_and(select_min, select_max)
                        if np.any(select_cross):
                            spatial.grid.row.bounds
                            spatial.grid.row.bounds = None
                            spatial.grid.col.bounds
                            spatial.grid.col.bounds = None
                            bounds_cross_meridian = True
                        else:
                            select = ref > 180
                            ref[select] -= 360

                # attempt to wrap the grid corners if they exist
                try:
                    ref = spatial.grid.corners.data
                    if bounds_cross_meridian:
                        spatial.grid._corners = None
                    else:
                        bounds_min = np.min(ref[1], axis=2)
                        bounds_max = np.max(ref[1], axis=2)
                        select_min = bounds_min <= 180
                        select_max = bounds_max > 180
                        select_cross = np.logical_and(select_min, select_max)
                        if np.any(select_cross):
                            spatial.grid._corners = None
                        else:
                            select = ref[1] > 180
                            ref[1][select] -= 360
                except CornersUnavailable:
                    pass
        else:
            ocgis_lh(exc=SpatialWrappingError('Data does not have a 0 to 360 coordinate system.'))

    def _get_to_wrap_(self,spatial):
        ret = []
        ret.append(spatial.geom.point)
        try:
            ret.append(spatial.geom.polygon)
        except ImproperPolygonBoundsError:
            pass
        return(ret)

    @staticmethod
    def _place_prime_meridian_array_(arr):
        """
        Replace any 180 degree values with the value of :attribute:`ocgis.constants.prime_meridian`.

        :param arr: The target array to modify inplace.
        :type arr: :class:`numpy.array`
        :rtype: boolean ;class:`numpy.array`
        """
        from ocgis import constants

        # find the values that are 180
        select = arr == 180
        # replace the values that are 180 with the constant value
        np.place(arr, select, constants.prime_meridian)
        # return the mask used for the replacement
        return select

    
class Spherical(CoordinateReferenceSystem, WrappableCoordinateReferenceSystem):
    """
    A spherical model of the Earth's surface with equivalent semi-major and semi-minor axes.

    :param semi_major_axis: The radius of the spherical model. The default value is taken from the PROJ.4 (v4.8.0)
     source code (src/pj_ellps.c).
    :type semi_major_axis: float
    """
    
    def __init__(self, semi_major_axis=6370997.0):
        value = {'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0', 'no_defs': '', 'a': semi_major_axis, 'b': semi_major_axis}
        CoordinateReferenceSystem.__init__(self, value=value)
        self.major_axis = semi_major_axis


class WGS84(CoordinateReferenceSystem, WrappableCoordinateReferenceSystem):
    """
    A representation of the Earth using the WGS84 datum (i.e. EPSG code 4326).
    """

    def __init__(self):
        CoordinateReferenceSystem.__init__(self, epsg=4326)


class CFCoordinateReferenceSystem(CoordinateReferenceSystem):
    __metaclass__ = abc.ABCMeta
    
    ## if False, no attempt to read projection coordinates will be made. they
    ## will be set to a None default.
    _find_projection_coordinates = True
    
    def __init__(self,**kwds):
        self.projection_x_coordinate = kwds.pop('projection_x_coordinate',None)
        self.projection_y_coordinate = kwds.pop('projection_y_coordinate',None)
        
        check_keys = kwds.keys()
        for key in kwds.keys():
            check_keys.remove(key)
        if len(check_keys) > 0:
            exc = ValueError('The keyword parameter(s) "{0}" was/were not provided.'.format(check_keys))
            ocgis_lh(exc=exc,logger='crs')
        
        self.map_parameters_values = kwds
        crs = {'proj':self.proj_name}
        for k in self.map_parameters.keys():
            if k in self.iterable_parameters:
                v = getattr(self,self.iterable_parameters[k])(kwds[k])
                crs.update(v)
            else:
                crs.update({self.map_parameters[k]:kwds[k]})
                
        super(CFCoordinateReferenceSystem,self).__init__(value=crs)
            
    @abc.abstractproperty
    def grid_mapping_name(self): str
    
    @abc.abstractproperty
    def iterable_parameters(self): dict
    
    @abc.abstractproperty
    def map_parameters(self): dict
    
    @abc.abstractproperty
    def proj_name(self): str
    
    def format_standard_parallel(self,value):
        if isinstance(value,np.ndarray):
            value = value.tolist()
            
        ret = {}
        try:
            it = iter(value)
        except TypeError:
            it = [value]
        for ii,v in enumerate(it,start=1):
            ret.update({self.map_parameters['standard_parallel'].format(ii):v})
        return(ret)
    
    @classmethod
    def load_from_metadata(cls,var,meta):

        def _get_projection_coordinate_(target,meta):
            key = 'projection_{0}_coordinate'.format(target)
            for k,v in meta['variables'].iteritems():
                if 'standard_name' in v['attrs']:
                    if v['attrs']['standard_name'] == key:
                        return(k)
            ocgis_lh(logger='crs',exc=ProjectionCoordinateNotFound(key))
            
        r_var = meta['variables'][var]
        try:
            ## look for the grid_mapping attribute on the target variable
            r_grid_mapping = meta['variables'][r_var['attrs']['grid_mapping']]
        except KeyError:
            raise(ProjectionDoesNotMatch)
        try:
            grid_mapping_name = r_grid_mapping['attrs']['grid_mapping_name']
        except KeyError:
            ocgis_lh(logger='crs',level=logging.WARN,msg='"grid_mapping" variable "{0}" does not have a "grid_mapping_name" attribute'.format(r_grid_mapping['name']))
            raise(ProjectionDoesNotMatch)
        if grid_mapping_name != cls.grid_mapping_name:
            raise(ProjectionDoesNotMatch)
        
        ## get the projection coordinates if not turned off by class attribute.
        if cls._find_projection_coordinates:
            pc_x,pc_y = [_get_projection_coordinate_(target,meta) for target in ['x','y']]
        else:
            pc_x,pc_y = None,None
        
        ## this variable name is used by the netCDF converter
        meta['grid_mapping_variable_name'] = r_grid_mapping['name']
        
        kwds = r_grid_mapping['attrs'].copy()
        kwds.pop('grid_mapping_name',None)
        kwds['projection_x_coordinate'] = pc_x
        kwds['projection_y_coordinate'] = pc_y
        
        cls._load_from_metadata_finalize_(kwds,var,meta)
        
        return(cls(**kwds))
    
    def write_to_rootgrp(self,rootgrp,meta):
        name = meta['grid_mapping_variable_name']
        crs = rootgrp.createVariable(name,meta['variables'][name]['dtype'])
        attrs = meta['variables'][name]['attrs']
        crs.setncatts(attrs)
    
    @classmethod
    def _load_from_metadata_finalize_(cls,kwds,var,meta):
        pass


class CFWGS84(WGS84,CFCoordinateReferenceSystem,):
    grid_mapping_name = 'latitude_longitude'
    iterable_parameters = None
    map_parameters = None
    proj_name = None
    
    def __init__(self):
        WGS84.__init__(self)
    
    @classmethod
    def load_from_metadata(cls,var,meta):
        try:
            r_grid_mapping = meta['variables'][var]['attrs']['grid_mapping']
            if r_grid_mapping == cls.grid_mapping_name:
                return(cls())
            else:
                raise(ProjectionDoesNotMatch)
        except KeyError:
            raise(ProjectionDoesNotMatch)
    
    
class CFAlbersEqualArea(CFCoordinateReferenceSystem):
    grid_mapping_name = 'albers_conical_equal_area'
    iterable_parameters = {'standard_parallel':'format_standard_parallel'}
    map_parameters = {'standard_parallel':'lat_{0}',
                      'longitude_of_central_meridian':'lon_0',
                      'latitude_of_projection_origin':'lat_0',
                      'false_easting':'x_0',
                      'false_northing':'y_0'}
    proj_name = 'aea'


class CFLambertConformal(CFCoordinateReferenceSystem):
    grid_mapping_name = 'lambert_conformal_conic'
    iterable_parameters = {'standard_parallel':'format_standard_parallel'}
    map_parameters = {'standard_parallel':'lat_{0}',
                      'longitude_of_central_meridian':'lon_0',
                      'latitude_of_projection_origin':'lat_0',
                      'false_easting':'x_0',
                      'false_northing':'y_0',
                      'units':'units'}
    proj_name = 'lcc'
    
    @classmethod
    def _load_from_metadata_finalize_(cls,kwds,var,meta):
        kwds['units'] = meta['variables'][kwds['projection_x_coordinate']]['attrs'].get('units')
        
        
class CFPolarStereographic(CFCoordinateReferenceSystem):
    grid_mapping_name = 'polar_stereographic'
    map_parameters = {'standard_parallel':'lat_ts',
                      'latitude_of_projection_origin':'lat_0',
                      'straight_vertical_longitude_from_pole':'lon_0',
                      'false_easting':'x_0',
                      'false_northing':'y_0',
                      'scale_factor':'k_0'}
    proj_name = 'stere'
    iterable_parameters = {}
    
    def __init__(self,*args,**kwds):
        if 'scale_factor' not in kwds:
            kwds['scale_factor'] = 1.0
        super(CFPolarStereographic,self).__init__(*args,**kwds)
    
    
class CFNarccapObliqueMercator(CFCoordinateReferenceSystem):
    grid_mapping_name = 'transverse_mercator'
    map_parameters = {'latitude_of_projection_origin':'lat_0',
                      'longitude_of_central_meridian':'lonc',
                      'scale_factor_at_central_meridian':'k_0',
                      'false_easting':'x_0',
                      'false_northing':'y_0',
                      'alpha':'alpha'}
    proj_name = 'omerc'
    iterable_parameters = {}
    
    def __init__(self,*args,**kwds):
        if 'alpha' not in kwds:
            kwds['alpha'] = 360
        super(CFNarccapObliqueMercator,self).__init__(*args,**kwds)


class CFRotatedPole(CFCoordinateReferenceSystem):
    grid_mapping_name = 'rotated_latitude_longitude'
    map_parameters = {'grid_north_pole_longitude': None,
                      'grid_north_pole_latitude': None}
    proj_name = 'omerc'
    iterable_parameters = {}
    _template = '+proj=ob_tran +o_proj=latlon +o_lon_p={lon_pole} +o_lat_p={lat_pole} +lon_0=180'
    _find_projection_coordinates = False

    def __init__(self, *args, **kwds):
        super(CFRotatedPole, self).__init__(*args, **kwds)

        # this is the transformation string used in the proj operation
        self._trans_proj = self._template.format(lon_pole=kwds['grid_north_pole_longitude'],
                                                 lat_pole=kwds['grid_north_pole_latitude'])

        # holds metadata and previous state information for inverse transformations
        self._inverse_state = {}

    def get_rotated_pole_transformation(self, spatial, inverse=False):
        """
        :type spatial: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :param bool inverse: If ``True``, this is an inverse transformation.
        :rtype: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        new_spatial = copy(spatial)
        new_spatial._geom = None

        try:
            rc_original = {'row': {'name': spatial.grid.row.name,
                                   'meta': spatial.grid.row.meta},
                           'col': {'name': spatial.grid.col.name,
                                   'meta': spatial.grid.col.meta}}
        # a previously transformed rotated pole spatial dimension will not have row and columns. these should be
        # available in the state dictionary
        except AttributeError:
            rc_original = self._inverse_state['rc_original']

        # if this metadata information is not stored, put in the state dictionary to use for inverse transformations
        if 'rc_original' not in self._inverse_state:
            self._inverse_state['rc_original'] = rc_original

        new_spatial.grid = self._get_rotated_pole_transformation_for_grid_(new_spatial.grid, inverse=inverse,
                                                                           rc_original=rc_original)

        # ensure the masks are updated appropriately
        new_spatial.grid.value.mask = spatial.grid.value.mask.copy()
        new_spatial.grid.uid.mask = spatial.grid.uid.mask.copy()

        # the crs has been transformed, so updated accordingly
        if inverse:
            new_spatial.crs = deepcopy(self)
        else:
            new_spatial.crs = CFWGS84()

        return new_spatial

    def _get_rotated_pole_transformation_for_grid_(self, grid, inverse=False, rc_original=None):
        """
        http://osgeo-org.1560.x6.nabble.com/Rotated-pole-coordinate-system-a-howto-td3885700.html

        :param :class:`ocgis.interface.base.dimension.spatial.SpatialGridDimension` grid:
        :param bool inverse: If ``True``, this is an inverse transformation.
        :param dict rc_original: Contains original metadata information for the row and
         column dimensions.
        :returns: :class:`ocgis.interface.base.dimension.spatial.SpatialGridDimension`
        """

        import csv
        import subprocess

        class ProjDialect(csv.excel):
            lineterminator = '\n'
            delimiter = '\t'

        f = tempfile.NamedTemporaryFile()
        writer = csv.writer(f, dialect=ProjDialect)
        new_mask = grid.value.mask.copy()

        if inverse:
            _row = grid.value[0, :, :].data
            _col = grid.value[1, :, :].data
            shp = (_row.shape[0], _col.shape[1])

            def _itr_writer_(_row, _col):
                for row_idx, col_idx in itertools.product(range(_row.shape[0]), range(_row.shape[1])):
                    yield (_col[row_idx, col_idx], _row[row_idx, col_idx])
        else:
            _row = grid.row.value
            _col = grid.col.value
            shp = (_row.shape[0], _col.shape[0])

            def _itr_writer_(row, col):
                for row_idx, col_idx in itertools.product(range(_row.shape[0]), range(_col.shape[0])):
                    yield (_col[col_idx], _row[row_idx])

        for xy in _itr_writer_(_row, _col):
            writer.writerow(xy)
        f.flush()
        cmd = self._trans_proj.split(' ')
        cmd.append(f.name)

        if inverse:
            program = 'invproj'
        else:
            program = 'proj'

        cmd = [program, '-f', '"%.6f"', '-m', '57.2957795130823'] + cmd
        capture = subprocess.check_output(cmd)
        f.close()
        coords = capture.split('\n')
        new_coords = []

        for ii, coord in enumerate(coords):
            coord = coord.replace('"', '')
            coord = coord.split('\t')
            try:
                coord = map(float, coord)
            # # likely empty string
            except ValueError:
                if coord[0] == '':
                    continue
                else:
                    raise
            new_coords.append(coord)

        new_coords = np.array(new_coords)
        new_row = new_coords[:, 1].reshape(*shp)
        new_col = new_coords[:, 0].reshape(*shp)

        new_grid = copy(grid)
        # # reset geometries
        new_grid._geom = None
        if inverse:
            from ocgis.interface.base.dimension.base import VectorDimension

            dict_row = self._get_meta_name_(rc_original, 'row')
            dict_col = self._get_meta_name_(rc_original, 'col')

            new_row = new_row[:, 0]
            new_col = new_col[0, :]
            new_grid.row = VectorDimension(value=new_row, name=dict_row['name'],
                                           meta=dict_row['meta'])
            new_grid.col = VectorDimension(value=new_col, name=dict_col['name'],
                                           meta=dict_col['meta'])
            new_col, new_row = np.meshgrid(new_col, new_row)
        else:
            new_grid._row_src_idx = new_grid.row._src_idx
            new_grid._col_src_idx = new_grid.col._src_idx
            new_grid.row = None
            new_grid.col = None

        new_value = np.zeros([2] + list(new_row.shape))
        new_value = np.ma.array(new_value, mask=new_mask)
        new_value[0, :, :] = new_row
        new_value[1, :, :] = new_col
        new_grid._value = new_value

        return new_grid

    @staticmethod
    def _get_meta_name_(rc_original, key):
        try:
            meta = rc_original[key]['meta']
            name = rc_original[key]['name']
        except TypeError:
            if rc_original is None:
                meta = None
                name = None
            else:
                raise
        return {'meta': meta, 'name': name}
