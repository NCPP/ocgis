import unittest
import itertools
from shapely.geometry import Point
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84,\
    CFAlbersEqualArea, CFLambertConformal, CFRotatedPole, CFWGS84
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGridDimension,\
    SpatialDimension
from ocgis.exc import SpatialWrappingError
from ocgis.test.base import TestBase
import numpy as np
from copy import deepcopy
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.util.helpers import get_temp_path, write_geom_dict
import netCDF4 as nc
from ocgis.interface.metadata import NcMetadata
import ocgis


class TestCoordinateReferenceSystem(TestBase):

    def test_constructor(self):
        crs = CoordinateReferenceSystem(epsg=4326)
        self.assertEqual(crs.sr.ExportToProj4(),'+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs ')
        
        crs2 = CoordinateReferenceSystem(prjs='+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs ')
        self.assertTrue(crs == crs2)
        self.assertFalse(crs != crs2)
        self.assertFalse(crs == None)   
        self.assertFalse(None == crs)

class TestWGS84(TestBase):

    def test_wrap_unwrap_with_mask(self):
        """Test wrapped and unwrapped geometries with a mask ensuring that masked values are wrapped and unwrapped."""

        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23])
        ret = ops.execute()
        sdim = ret[23]['tas'].spatial
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162], [37.67309213352349, 37.67309213352349, 37.67309213352349], [40.4636506825932, 40.4636506825932, 40.4636506825932], [43.254197169829105, 43.254197169829105, 43.254197169829105]], [[-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125]]], dtype=sdim.grid.value.dtype)
        self.assertNumpyAll(actual, sdim.grid.value.data)

        WGS84().unwrap(sdim)
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162], [37.67309213352349, 37.67309213352349, 37.67309213352349], [40.4636506825932, 40.4636506825932, 40.4636506825932], [43.254197169829105, 43.254197169829105, 43.254197169829105]], [[239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875]]], dtype=sdim.grid.value.dtype)
        self.assertNumpyAll(actual, sdim.grid.value.data)

    def test_wrap_normal_differing_data_types(self):
        row = VectorDimension(value=40.,bounds=[38.,42.])
        col = VectorDimension(value=0,bounds=[-1,1])
        with self.assertRaises(ValueError):
            SpatialGridDimension(row=row,col=col).value
            
    def test_wrap_normal(self):
        row = VectorDimension(value=40.,bounds=[38.,42.])
        col = VectorDimension(value=0.,bounds=[-1.,1.])
        grid = SpatialGridDimension(row=row,col=col)   
        self.assertEqual(grid.resolution,3.0)
        sdim = SpatialDimension(grid=grid,crs=WGS84())
        with self.assertRaises(SpatialWrappingError):
            sdim.crs.wrap(sdim)
        sdim.crs.unwrap(sdim)
        self.assertNotEqual(sdim.grid,None)
        self.assertNumpyAll(sdim.grid.value,np.ma.array(data=[[[40.0]],[[0.0]]],mask=[[[False]],[[False]]],))
    
    def test_wrap_360(self):
        row = VectorDimension(value=40.,bounds=[38.,42.])
        col = VectorDimension(value=181.5,bounds=[181.,182.])
        grid = SpatialGridDimension(row=row,col=col)
        self.assertEqual(grid.value[1,0,0],181.5)
        sdim = SpatialDimension(grid=grid,crs=WGS84())
        orig_sdim = deepcopy(sdim)
        orig_grid = deepcopy(sdim.grid)
        sdim.crs.wrap(sdim)
        self.assertNumpyAll(np.array(sdim.geom.point.value[0,0]),np.array([-178.5,40.]))
        self.assertEqual(sdim.geom.polygon.value[0,0].bounds,(-179.0,38.0,-178.0,42.0))
        self.assertNumpyNotAll(orig_grid.value,sdim.grid.value)
        sdim.crs.unwrap(sdim)
        sdim.set_grid_bounds_from_geometry()
        orig_sdim.set_grid_bounds_from_geometry()
        to_test = ([sdim.grid.value,orig_sdim.grid.value],[sdim.grid.bounds,orig_sdim.grid.bounds])
        for tt in to_test:
            self.assertNumpyAll(*tt)
    
    def test_wrap_360_cross_axis(self):
        row = VectorDimension(value=40,bounds=[38,42])
        col = VectorDimension(value=180,bounds=[179,181])
        grid = SpatialGridDimension(row=row,col=col)
        sdim = SpatialDimension(grid=grid,crs=WGS84())
        orig_sdim = deepcopy(sdim)
        sdim.crs.wrap(sdim)
        self.assertIsInstance(sdim.geom.polygon.value[0,0],MultiPolygon)
        sdim.crs.unwrap(sdim)
        self.assertEqual(orig_sdim.geom.polygon.value[0,0].bounds,sdim.geom.polygon.value[0,0].bounds)
        for target in ['point','polygon']:
            path = get_temp_path(name=target,suffix='.shp',wd=self._test_dir)
            sdim.write_fiona(path,target)
            
            
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
        uri = self.test_data.get_uri('narccap_wrfg')
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
        rd = self.test_data.get_rd('rotated_pole_ichec')
        self.assertIsInstance(rd.get().spatial.crs, CFRotatedPole)

    def test_equal(self):
        rd = self.test_data.get_rd('rotated_pole_ichec')
        rd2 = deepcopy(rd)
        self.assertEqual(rd.get().spatial.crs, rd2.get().spatial.crs)

    def test_in_operations(self):
        rd = self.test_data.get_rd('rotated_pole_ichec')
        rd2 = deepcopy(rd)
        rd2.alias = 'tas2'
        # # these projections are equivalent so it is okay to write them to a
        ## common output file
        ops = ocgis.OcgOperations(dataset=[rd, rd2], output_format='csv', snippet=True)
        ops.execute()

    def test_get_rotated_pole_transformation(self):
        """Test SpatialDimension objects are appropriately transformed."""

        rd = self.test_data.get_rd('rotated_pole_ichec')
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
