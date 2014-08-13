import unittest
import fiona
from ocgis.interface.base.crs import WGS84
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGeometryPolygonDimension, \
    SpatialGeometryPointDimension
from ocgis.util.shp_cabinet import ShpCabinet, ShpCabinetIterator
import ocgis
from osgeo import ogr
from ocgis.test.base import TestBase
import os
import shutil
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon


class TestShpCabinetIterator(TestBase):

    def test_as_spatial_dimension(self):
        """Test iteration returned as SpatialDimension objects."""

        select_ugid = [16, 17, 51]
        sci = ShpCabinetIterator(key='state_boundaries', select_ugid=select_ugid, as_spatial_dimension=True)

        for _ in range(2):
            ugids = []
            for sdim in sci:
                self.assertIsInstance(sdim, SpatialDimension)
                self.assertEqual(sdim.shape, (1, 1))
                self.assertIsInstance(sdim.geom.get_highest_order_abstraction(), SpatialGeometryPolygonDimension)
                self.assertEqual(sdim.properties[0]['UGID'], sdim.uid[0, 0])
                self.assertEqual(sdim.properties.dtype.names, ('UGID', 'STATE_FIPS', 'ID', 'STATE_NAME', 'STATE_ABBR'))
                self.assertEqual(sdim.crs, WGS84())

                for prop in sdim.properties[0]:
                    self.assertNotEqual(prop, None)

                ugids.append(sdim.uid[0, 0])
            self.assertEqual(ugids, select_ugid)

    def test_as_spatial_dimension_points(self):
        """Test SpatialDimension iteration with a point shapefile as source."""

        sci = ShpCabinetIterator(key='qed_city_centroids', as_spatial_dimension=True)
        for sdim in sci:
            self.assertIsInstance(sdim.geom.get_highest_order_abstraction(), SpatialGeometryPointDimension)

    def test_select_ugids_absent_raises_exception(self):
        sci = ShpCabinetIterator(key='state_boundaries',select_ugid=[999])
        with self.assertRaises(ValueError):
            list(sci)
            
        ops = ocgis.OcgOperations(dataset=self.test_data.get_rd('cancm4_tas'),
                                  geom='state_boundaries',
                                  select_ugid=[9999])
        with self.assertRaises(ValueError):
            ops.execute()
    
    def test_iteration_no_geoms(self):
        sci = ShpCabinetIterator(key='state_boundaries',load_geoms=False)
        for geom in sci:
            self.assertNotIn('geom',geom)
    
    def test_len(self):
        path = ShpCabinet().get_shp_path('state_boundaries')
        sci = ShpCabinetIterator(path=path)
        self.assertEqual(len(sci),51)
        sci = ShpCabinetIterator(path=path,select_ugid=[16,19])
        self.assertEqual(len(sci),2)
    
    def test_iteration_by_path(self):
        ## test that a shapefile may be retrieved by passing a full path to the
        ## file
        path = ShpCabinet().get_shp_path('state_boundaries')
        ocgis.env.DIR_SHPCABINET = None
        sci = ShpCabinetIterator(path=path)
        self.assertEqual(len(list(sci)),51)
        for geom in sci:
            self.assertIn(type(geom['geom']),(Polygon,MultiPolygon))

    def test_iteration_by_path_with_bad_path(self):
        ## if the path does not exist on the filesystem, then an exception should
        ## be raised
        ocgis.env.DIR_SHPCABINET = None
        sci = ShpCabinetIterator(path='/foo/foo/foo/foo/foo')
        with self.assertRaises(RuntimeError):
            list(sci)
            
    def test_key_used_before_path(self):
        ## the key always takes preference over the path
        sci = ShpCabinetIterator(key='state_boundaries',path='/foo/foo/foo/foo/foo')
        self.assertEqual(len(list(sci)),51)
            

class TestShpCabinet(TestBase):

    def test_number_in_shapefile_name(self):
        """Test number in shapefile name."""

        sc = ShpCabinet()
        path = sc.get_shp_path('state_boundaries')
        out_path = os.path.join(self._test_dir, '51_states.shp')
        with fiona.open(path) as source:
            with fiona.open(out_path, mode='w', driver='ESRI Shapefile', schema=source.meta['schema'], crs=source.meta['crs']) as sink:
                for record in source:
                    sink.write(record)
        ret = list(ShpCabinetIterator(select_ugid=[23], path=out_path))
        self.assertEqual(len(ret), 1)

    def test_iter_geoms_select_ugid_is_sorted(self):
        sc = ShpCabinet()
        with self.assertRaises(ValueError):
            list(sc.iter_geoms('state_boundaries',select_ugid=[23,18]))
    
    def test_iter_geoms_no_load_geoms(self):
        sc = ShpCabinet()
        it = sc.iter_geoms('state_boundaries',load_geoms=False)
        geoms = list(it)
        self.assertEqual(len(geoms),51)
        self.assertEqual(geoms[12]['properties']['STATE_NAME'],'New Hampshire')
        for geom in geoms:
            self.assertNotIn('geom',geom)
    
    def test_iter_geoms(self):
        sc = ShpCabinet()
        it = sc.iter_geoms('state_boundaries')
        geoms = list(it)
        self.assertEqual(len(geoms),51)
        self.assertEqual(geoms[12]['properties']['STATE_NAME'],'New Hampshire')
        for geom in geoms:
            self.assertIn(type(geom['geom']),(Polygon,MultiPolygon))
        
    def test_iter_geoms_select_ugid(self):
        sc = ShpCabinet()
        it = sc.iter_geoms('state_boundaries',select_ugid=[13])
        geoms = list(it)
        self.assertEqual(len(geoms),1)
        self.assertEqual(geoms[0]['properties']['STATE_NAME'],'New Hampshire')
    
    def test_sql_subset(self):
        sc = ShpCabinet()
        path = sc.get_shp_path('state_boundaries')
        ds = ogr.Open(path)
        ret = ds.ExecuteSQL('select * from state_boundaries where state_name = "New Jersey"')
        ret.ResetReading()
        self.assertEqual(len(ret),1)
    
    def test_bad_path(self):
        bp = '/a/bad/location'
        with self.assertRaises(ValueError):
            list(ShpCabinet(bp).iter_geoms('state_boundaries'))
            
    def test_none_path(self):
        try:
            ocgis.env.DIR_SHPCABINET = None
            with self.assertRaises(ValueError):
                list(ShpCabinet().iter_geoms('state_boundaries'))
        finally:
            ocgis.env.reset()

    def test_get_keys(self,dir_shpcabinet=None):
        ocgis.env.DIR_SHPCABINET = dir_shpcabinet or ocgis.env.DIR_SHPCABINET
        sc = ShpCabinet()
        ret = sc.keys()
        target_keys = ['state_boundaries','world_countries']
        self.assertEqual(len(set(target_keys).intersection(set(ret))),len(target_keys))
            
    def test_shapefiles_not_in_folders(self):
        for dirpath,dirnames,filenames in os.walk(ocgis.env.DIR_SHPCABINET):
            for filename in filenames:
                if filename.startswith('state_boundaries') or filename.startswith('world_countries'):
                    dst = os.path.join(self._test_dir,filename)
                    src = os.path.join(dirpath,filename)
                    shutil.copy2(src,dst)
        self.test_get_keys(dir_shpcabinet=self._test_dir)
        
        sc = ShpCabinet(path=self._test_dir)
        path = sc.get_shp_path('world_countries')
        self.assertEqual(path,os.path.join(self._test_dir,'world_countries.shp'))
            
            
if __name__ == '__main__':
    unittest.main()
    