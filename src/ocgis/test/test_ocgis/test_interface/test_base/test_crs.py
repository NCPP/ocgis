import unittest
import itertools
from shapely.geometry import Point, MultiPoint
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84,\
    CFAlbersEqualArea, CFLambertConformal, CFRotatedPole, CFWGS84, Spherical, WrappableCoordinateReferenceSystem
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGridDimension,\
    SpatialDimension
from ocgis.exc import SpatialWrappingError, CornersUnavailable
from ocgis.test.base import TestBase
import numpy as np
from copy import deepcopy
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.util.helpers import get_temp_path, write_geom_dict, make_poly
import netCDF4 as nc
from ocgis.interface.metadata import NcMetadata
import ocgis
from ocgis.util.itester import itr_products_keywords
from ocgis import constants


class TestCoordinateReferenceSystem(TestBase):

    def test_init(self):
        keywords = dict(value=[None, {'no_defs': True, 'ellps': 'WGS84', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'}],
                        epsg=[None, 4326],
                        proj4=[None, '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs '])

        prev_crs = None
        for k in itr_products_keywords(keywords):
            try:
                crs = CoordinateReferenceSystem(**k)
            except ValueError:
                if all([ii is None for ii in k.values()]):
                    continue
                else:
                    raise
            self.assertEqual(crs.proj4, '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs ')
            self.assertDictEqual(crs.value, {'no_defs': True, 'ellps': 'WGS84', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'})
            if prev_crs is not None:
                self.assertEqual(crs, prev_crs)
            prev_crs = deepcopy(crs)

    def test_ne(self):
        crs1 = CoordinateReferenceSystem(epsg=4326)
        crs2 = CoordinateReferenceSystem(epsg=2136)

        self.assertNotEqual(crs1, crs2)
        self.assertNotEqual(crs2, crs1)
        self.assertNotEqual(crs1, None)
        self.assertNotEqual(None, crs1)

        # try nonetype and string
        self.assertNotEqual(None, crs1)
        self.assertNotEqual('input', crs1)


class TestSpherical(TestBase):

    def test_init(self):
        crs = Spherical()
        self.assertDictEqual(crs.value, {'a': 6370997, 'no_defs': True, 'b': 6370997, 'proj': 'longlat',
                                         'towgs84': '0,0,0,0,0,0,0'})

        crs = Spherical(semi_major_axis=6370998.1)
        self.assertDictEqual(crs.value, {'a': 6370998.1, 'no_defs': True, 'b': 6370998.1, 'proj': 'longlat',
                                         'towgs84': '0,0,0,0,0,0,0'})

    def test_get_is_360_geometries(self):
        bounds = (5.869442939758301, 47.28110122680663, 15.038049697875975, 54.91740036010742)
        poly = make_poly((bounds[1], bounds[3]), (bounds[0], bounds[2]))
        record_poly = {'geom': poly, 'properties': {'UGID': 1}}
        record_point = {'geom': poly.centroid, 'properties': {'UGID': 1}}
        for record in [record_poly, record_point]:
            sdim = SpatialDimension.from_records([record])
            self.assertFalse(Spherical.get_is_360(sdim))

    def test_get_is_360_grid(self):
        # perform test with small grid falling between 0 and 180.
        row = VectorDimension(value=[0, 40])
        col = VectorDimension(value=[0, 170])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid)
        # no crs for the spatial dimension, hence wrapping will fail.
        with self.assertRaises(SpatialWrappingError):
            self.assertIsNone(sdim.crs)
            Spherical.get_is_360(sdim)
        sdim.crs = Spherical()
        self.assertFalse(Spherical.get_is_360(sdim))

    def test_place_prime_meridian_array(self):
        arr = np.array([123, 180, 200, 180], dtype=float)
        ret = Spherical._place_prime_meridian_array_(arr)
        self.assertNumpyAll(ret, np.array([False, True, False, True]))
        self.assertNumpyAll(arr, np.array([123., constants.prime_meridian, 200., constants.prime_meridian]))

    def test_wrap_unwrap_with_mask(self):
        """Test wrapped and unwrapped geometries with a mask ensuring that masked values are wrapped and unwrapped."""

        rd = self.test_data_nc.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23])
        ret = ops.execute()
        sdim = ret[23]['tas'].spatial
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162], [37.67309213352349, 37.67309213352349, 37.67309213352349], [40.4636506825932, 40.4636506825932, 40.4636506825932], [43.254197169829105, 43.254197169829105, 43.254197169829105]], [[-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125]]], dtype=sdim.grid.value.dtype)
        self.assertNumpyAll(actual, sdim.grid.value.data)

        Spherical().unwrap(sdim)
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162], [37.67309213352349, 37.67309213352349, 37.67309213352349], [40.4636506825932, 40.4636506825932, 40.4636506825932], [43.254197169829105, 43.254197169829105, 43.254197169829105]], [[239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875]]], dtype=sdim.grid.value.dtype)
        self.assertNumpyAll(actual, sdim.grid.value.data)

    def test_wrap_normal(self):
        """Test exception thrown when attempting to wrap already wrapped coordinate values."""

        row = VectorDimension(value=40., bounds=[38., 42.])
        col = VectorDimension(value=0., bounds=[-1., 1.])
        grid = SpatialGridDimension(row=row, col=col)
        self.assertEqual(grid.resolution, 3.0)
        sdim = SpatialDimension(grid=grid, crs=Spherical())
        with self.assertRaises(SpatialWrappingError):
            sdim.crs.wrap(sdim)
        sdim.crs.unwrap(sdim)
        self.assertNotEqual(sdim.grid, None)
        self.assertNumpyAll(sdim.grid.value, np.ma.array(data=[[[40.0]], [[0.0]]], mask=[[[False]], [[False]]], ))
    
    def test_wrap_360(self):
        """Test wrapping."""

        row = VectorDimension(value=40., bounds=[38., 42.])
        col = VectorDimension(value=181.5, bounds=[181., 182.])
        grid = SpatialGridDimension(row=row, col=col)
        self.assertEqual(grid.value[1, 0, 0], 181.5)
        sdim = SpatialDimension(grid=grid, crs=WGS84())
        orig_sdim = deepcopy(sdim)
        orig_grid = deepcopy(sdim.grid)
        sdim.crs.wrap(sdim)
        self.assertNumpyAll(np.array(sdim.geom.point.value[0, 0]), np.array([-178.5, 40.]))
        self.assertEqual(sdim.geom.polygon.value[0, 0].bounds, (-179.0, 38.0, -178.0, 42.0))
        self.assertNumpyNotAll(orig_grid.value, sdim.grid.value)
        sdim.crs.unwrap(sdim)
        to_test = ([sdim.grid.value, orig_sdim.grid.value], [sdim.grid.corners, orig_sdim.grid.corners])
        for tt in to_test:
            self.assertNumpyAll(*tt)
    
    def test_wrap_360_prime_meridian(self):
        """Test wrapping with bounds interacting with the prime meridian."""

        def _get_sdim_(value, bounds):
            row1 = VectorDimension(value=40., bounds=[38., 42.])
            try:
                bounds = map(float, bounds)
            except TypeError:
                bounds = [map(float, b) for b in bounds]
            try:
                value = float(value)
            except TypeError:
                value = [float(v) for v in value]
            col1 = VectorDimension(value=value, bounds=bounds)
            grid1 = SpatialGridDimension(row=row1, col=col1)
            sdim1 = SpatialDimension(grid=grid1, crs=Spherical())
            return deepcopy(sdim1), sdim1

        # bounds values at the prime meridian of 180.
        orig, sdim = _get_sdim_(178, [176, 180.])
        # data does not have a verified 360 coordinate system
        with self.assertRaises(SpatialWrappingError):
            sdim.wrap()

        # bounds values on the other side of the prime meridian
        orig, sdim = _get_sdim_(182, [180, 184])
        sdim.wrap()
        self.assertIsNone(sdim.grid.col.bounds)
        self.assertIsNone(sdim.grid.row.bounds)
        with self.assertRaises(CornersUnavailable):
            sdim.grid.corners
        self.assertEqual(sdim.geom.polygon.value[0, 0][0].bounds, (-180.0, 38.0, -176.0, 42.0))
        self.assertNumpyAll(np.array(sdim.geom.point.value[0, 0]), np.array([-178., 40.]))

        # centroid directly on prime meridian
        orig, sdim = _get_sdim_(180, [178, 182])
        sdim.wrap()
        self.assertIsNone(sdim.grid.col.bounds)
        self.assertIsNone(sdim.grid.row.bounds)
        with self.assertRaises(CornersUnavailable):
            sdim.grid.corners
        self.assertEqual(sdim.geom.polygon.value[0, 0][0].bounds, (178.0, 38.0, 180.0, 42.0))
        self.assertEqual(sdim.geom.polygon.value[0, 0][1].bounds, (-180.0, 38.0, -178.0, 42.0))
        self.assertNumpyAll(np.array(sdim.geom.point.value[0, 0]), np.array([180., 40.]))

        # no row/column bounds but with corners
        orig, sdim = _get_sdim_([182, 186], [[180, 184], [184, 188]])
        sdim.grid.corners
        sdim.grid.row.bounds
        sdim.grid.row.bounds = None
        sdim.grid.col.bounds
        sdim.grid.col.bounds = None
        sdim.wrap()
        with self.assertRaises(CornersUnavailable):
            sdim.grid.corners

        # unwrap a wrapped spatial dimension making sure the unwrapped multipolygon bounds are the same as the wrapped
        # polygon bounds.
        row = VectorDimension(value=40, bounds=[38, 42])
        col = VectorDimension(value=180, bounds=[179, 181])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid, crs=Spherical())
        orig_sdim = deepcopy(sdim)
        sdim.crs.wrap(sdim)
        self.assertIsInstance(sdim.geom.polygon.value[0, 0], MultiPolygon)
        sdim.crs.unwrap(sdim)
        self.assertEqual(orig_sdim.geom.polygon.value[0, 0].bounds, sdim.geom.polygon.value[0, 0].bounds)

        # for target in ['point', 'polygon']:
        #     path = get_temp_path(name=target, suffix='.shp', wd=self.current_dir_output)
        #     sdim.write_fiona(path, target)
            

class TestWGS84(TestBase):

    def test_init(self):
        self.assertEqual(WGS84(), CoordinateReferenceSystem(epsg=4326))
        self.assertIsInstance(WGS84(), WrappableCoordinateReferenceSystem)
        self.assertNotIsInstance(WGS84(), Spherical)


class TestCFAlbersEqualArea(TestBase):
    
    def test_constructor(self):
        crs = CFAlbersEqualArea(standard_parallel=[29.5,45.5],longitude_of_central_meridian=-96,
                                latitude_of_projection_origin=37.5,false_easting=0,
                                false_northing=0)
        self.assertEqual(crs.value,{'lon_0': -96, 'ellps': 'WGS84', 'y_0': 0, 'no_defs': True, 'proj': 'aea', 'x_0': 0, 'units': 'm', 'lat_2': 45.5, 'lat_1': 29.5, 'lat_0': 37.5})
    
    def test_empty(self):
        with self.assertRaises(KeyError):
            CFAlbersEqualArea()
            
    def test_bad_parms(self):
        with self.assertRaises(KeyError):
            CFAlbersEqualArea(standard_parallel=[29.5,45.5],longitude_of_central_meridian=-96,
                              latitude_of_projection_origin=37.5,false_easting=0,
                              false_nothing=0)


class TestCFLambertConformalConic(TestBase):
    
    def test_load_from_metadata(self):
        uri = self.test_data_nc.get_uri('narccap_wrfg')
        ds = nc.Dataset(uri,'r')
        meta = NcMetadata(ds)
        crs = CFLambertConformal.load_from_metadata('pr',meta)
        self.assertEqual(crs.value,{'lon_0': -97, 'ellps': 'WGS84', 'y_0': 2700000, 'no_defs': True, 'proj': 'lcc', 'x_0': 3325000, 'units': 'm', 'lat_2': 60, 'lat_1': 30, 'lat_0': 47.5})
        self.assertIsInstance(crs,CFLambertConformal)
        self.assertEqual(['xc','yc'],[crs.projection_x_coordinate,crs.projection_y_coordinate])
        self.assertNumpyAll(np.array([ 30.,  60.]),crs.map_parameters_values.pop('standard_parallel'))
        self.assertEqual(crs.map_parameters_values,{u'latitude_of_projection_origin': 47.5, u'longitude_of_central_meridian': -97.0, u'false_easting': 3325000.0, u'false_northing': 2700000.0, 'units': u'm'})
        ds.close()
        
        
class TestCFRotatedPole(TestBase):

    def test_load_from_metadata(self):
        rd = self.test_data_nc.get_rd('rotated_pole_ichec')
        self.assertIsInstance(rd.get().spatial.crs, CFRotatedPole)

    def test_equal(self):
        rd = self.test_data_nc.get_rd('rotated_pole_ichec')
        rd2 = deepcopy(rd)
        self.assertEqual(rd.get().spatial.crs, rd2.get().spatial.crs)

    def test_in_operations(self):
        rd = self.test_data_nc.get_rd('rotated_pole_ichec')
        rd2 = deepcopy(rd)
        rd2.alias = 'tas2'
        # # these projections are equivalent so it is okay to write them to a
        ## common output file
        ops = ocgis.OcgOperations(dataset=[rd, rd2], output_format='csv', snippet=True)
        ops.execute()

    def test_get_rotated_pole_transformation(self):
        """Test SpatialDimension objects are appropriately transformed."""

        rd = self.test_data_nc.get_rd('rotated_pole_ichec')
        field = rd.get()
        field = field[:, 10:20, :, 40:55, 55:65]
        spatial = field.spatial
        self.assertIsNotNone(spatial._grid)

        # modify the mask to ensure it is appropriately updated and copied during the transformations
        spatial.grid.value.mask[:, 5, 6] = True
        spatial.grid.uid.mask[5, 6] = True
        spatial.assert_uniform_mask()

        self.assertIsNone(spatial._geom)
        spatial.geom
        self.assertIsNotNone(spatial._geom)
        new_spatial = field.spatial.crs.get_rotated_pole_transformation(spatial)
        original_crs = deepcopy(field.spatial.crs)
        self.assertIsInstance(new_spatial.crs, CFWGS84)
        self.assertIsNone(new_spatial._geom)
        new_spatial.geom
        self.assertIsNotNone(new_spatial._geom)

        self.assertNumpyNotAllClose(spatial.grid.value, new_spatial.grid.value)

        field_copy = deepcopy(field)
        self.assertIsNone(field_copy.variables['tas']._value)
        field_copy.spatial = new_spatial
        value = field_copy.variables['tas'].value
        self.assertIsNotNone(field_copy.variables['tas']._value)
        self.assertIsNone(field.variables['tas']._value)

        self.assertNumpyAll(field.variables['tas'].value, field_copy.variables['tas'].value)

        inverse_spatial = original_crs.get_rotated_pole_transformation(new_spatial, inverse=True)
        inverse_spatial.assert_uniform_mask()

        self.assertNumpyAll(inverse_spatial.uid, spatial.uid)
        self.assertNumpyAllClose(inverse_spatial.grid.row.value, spatial.grid.row.value)
        self.assertNumpyAllClose(inverse_spatial.grid.col.value, spatial.grid.col.value)
        self.assertDictEqual(spatial.grid.row.meta, inverse_spatial.grid.row.meta)
        self.assertEqual(spatial.grid.row.name, inverse_spatial.grid.row.name)
        self.assertDictEqual(spatial.grid.col.meta, inverse_spatial.grid.col.meta)
        self.assertEqual(spatial.grid.col.name, inverse_spatial.grid.col.name)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
