import os
import shutil

import fiona
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

import ocgis
from ocgis import env
from ocgis.interface.base.crs import WGS84
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGeometryPolygonDimension, \
    SpatialGeometryPointDimension
from ocgis.test.base import TestBase
from ocgis.test.base import attr
from ocgis.util.environment import ogr
from ocgis.util.geom_cabinet import GeomCabinet, GeomCabinetIterator, get_uid_from_properties

Layer = ogr.Layer


class Test(TestBase):
    def test_get_uid_from_properties(self):
        properties = {env.DEFAULT_GEOM_UID: 1, 'name': 'food', 'ID': 3}
        uid, add_uid = get_uid_from_properties(properties, None)
        self.assertFalse(add_uid)
        self.assertEqual(uid, env.DEFAULT_GEOM_UID)
        with self.assertRaises(ValueError):
            get_uid_from_properties(properties, 'DD')

        properties = {'ID': 5, 'name': 'scooter'}
        uid, add_uid = get_uid_from_properties(properties, None)
        self.assertEqual(uid, env.DEFAULT_GEOM_UID)
        self.assertTrue(add_uid)
        with self.assertRaises(ValueError):
            get_uid_from_properties(properties, 'name')


class TestGeomCabinetIterator(TestBase):
    def setUp(self):
        super(TestGeomCabinetIterator, self).setUp()
        self._original_dir_geomcabinet = env.DIR_GEOMCABINET
        path = os.path.join(self.path_bin, 'shp')
        env.DIR_GEOMCABINET = path

    def test_init(self):
        sci = GeomCabinetIterator(key='state_boundaries', uid='ID', as_spatial_dimension=True)
        self.assertIsNone(sci.select_sql_where)
        self.assertEqual(sci.uid, 'ID')
        for sdim in sci:
            self.assertEqual(sdim.name_uid, 'ID')
            break

        s = 'STATE_NAME = "Wisconsin"'
        sci = GeomCabinetIterator(key='state_boundaries', select_sql_where=s)
        self.assertEqual(sci.select_sql_where, s)

    def test_as_spatial_dimension(self):
        """Test iteration returned as SpatialDimension objects."""

        select_ugid = [16, 17, 51]
        sci = GeomCabinetIterator(key='state_boundaries', select_uid=select_ugid, as_spatial_dimension=True)

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

    @attr('data')
    def test_as_spatial_dimension_points(self):
        """Test SpatialDimension iteration with a point shapefile as source."""

        env.DIR_GEOMCABINET = self._original_dir_geomcabinet
        sci = GeomCabinetIterator(key='qed_city_centroids', as_spatial_dimension=True)
        for sdim in sci:
            self.assertIsInstance(sdim.geom.get_highest_order_abstraction(), SpatialGeometryPointDimension)

    def test_iter(self):
        # test with a select statement
        sci = GeomCabinetIterator(key='state_boundaries', select_sql_where='STATE_NAME in ("Wisconsin", "Vermont")')
        for row in sci:
            self.assertIn(row['properties']['STATE_NAME'], ("Wisconsin", "Vermont"))

    @attr('data')
    def test_select_ugids_absent_raises_exception(self):
        path = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        sci = GeomCabinetIterator(path=path, select_uid=[999])
        with self.assertRaises(ValueError):
            list(sci)

        ops = ocgis.OcgOperations(dataset=self.test_data.get_rd('cancm4_tas'),
                                  geom=path,
                                  select_ugid=[9999])
        with self.assertRaises(ValueError):
            ops.execute()

    def test_iteration_no_geoms(self):
        sci = GeomCabinetIterator(key='state_boundaries', load_geoms=False)
        for geom in sci:
            self.assertNotIn('geom', geom)

    def test_len(self):
        path = GeomCabinet().get_shp_path('state_boundaries')
        sci = GeomCabinetIterator(path=path)
        self.assertEqual(len(sci), 51)
        sci = GeomCabinetIterator(path=path, select_uid=[16, 19])
        self.assertEqual(len(sci), 2)

        sci = GeomCabinetIterator(key='state_boundaries', select_sql_where='STATE_NAME = "Vermont"')
        self.assertEqual(len(sci), 1)

    def test_iteration_by_path(self):
        # test that a shapefile may be retrieved by passing a full path to the file
        path = GeomCabinet().get_shp_path('state_boundaries')
        ocgis.env.DIR_GEOMCABINET = None
        sci = GeomCabinetIterator(path=path)
        self.assertEqual(len(list(sci)), 51)
        for geom in sci:
            self.assertIn(type(geom['geom']), (Polygon, MultiPolygon))

    def test_iteration_by_path_with_bad_path(self):
        # if the path does not exist on the filesystem, then an exception should be raised
        ocgis.env.DIR_GEOMCABINET = None
        sci = GeomCabinetIterator(path='/foo/foo/foo/foo/foo')
        with self.assertRaises(RuntimeError):
            list(sci)

    def test_key_used_before_path(self):
        # the key always takes preference over the path
        sci = GeomCabinetIterator(key='state_boundaries', path='/foo/foo/foo/foo/foo')
        self.assertEqual(len(list(sci)), 51)


class TestGeomCabinet(TestBase):
    def setUp(self):
        super(TestGeomCabinet, self).setUp()
        path = os.path.join(self.path_bin, 'shp')
        env.DIR_GEOMCABINET = path

    def test_init(self):
        bp = '/a/bad/location'
        with self.assertRaises(ValueError):
            cabinet = GeomCabinet(bp)
            list(cabinet.iter_geoms('state_boundaries'))

        try:
            ocgis.env.set_geomcabinet_path(None)
            with self.assertRaises(ValueError):
                list(GeomCabinet().iter_geoms('state_boundaries'))
        finally:
            ocgis.env.reset()

    def test_get_features_object(self):
        # test with a shapefile not having the default unique geometry identifier
        path = self.get_shapefile_path_with_no_ugid()
        keywords = dict(uid=[None, 'ID'],
                        select_uid=[None, [8, 11, 13]],
                        select_sql_where=[None, 'STATE_NAME = "Wisconsin"'])

        for k in self.iter_product_keywords(keywords):
            ds = ogr.Open(path)
            try:
                try:
                    obj = GeomCabinet._get_features_object_(ds, uid=k.uid, select_uid=k.select_uid,
                                                            select_sql_where=k.select_sql_where)
                except RuntimeError:
                    self.assertIsNone(k.uid)
                    self.assertIsNotNone(k.select_uid)
                    continue

                if k.select_sql_where is not None:
                    length = 1
                elif k.select_uid is not None:
                    length = 3
                else:
                    length = 11
                self.assertEqual(len(obj), length)
                self.assertIsInstance(obj, Layer)
            finally:
                ds.Destroy()

        # test on a shapefile having the default unique geometry identifier
        path = GeomCabinet().get_shp_path('state_boundaries')
        ds = ogr.Open(path)
        try:
            obj = GeomCabinet._get_features_object_(ds, select_uid=[8, 11, 13])
            self.assertEqual(len(obj), 3)
        finally:
            ds.Destroy()

    def test_get_features_object_select_sql_where(self):
        path = GeomCabinet().get_shp_path('state_boundaries')

        def _run_(s, func):
            try:
                ds = ogr.Open(path)
                obj = GeomCabinet._get_features_object_(ds, select_sql_where=s)
                func(obj)
            finally:
                ds.Destroy()

        s = 'STATE_NAME in ("Wisconsin", "Vermont")'

        def f(obj):
            self.assertEqual(len(obj), 2)
            self.assertAsSetEqual([ii.items()['STATE_NAME'] for ii in obj], ("Wisconsin", "Vermont"))

        _run_(s, f)

        s = 'STATE_NAME in ("Wisconsin", "Vermont") and STATE_ABBR in ("NV", "OH")'

        def f(obj):
            self.assertEqual(len(obj), 0)

        _run_(s, f)

        s = 'STATE_NAME in ("Wisconsin", "Vermont") or STATE_ABBR in ("NV", "OH")'

        def f(obj):
            self.assertEqual(len(obj), 4)

        _run_(s, f)

        s = 'STATE_NAMEE in ("Wisconsin", "Vermont")'
        with self.assertRaises(RuntimeError):
            _run_(s, lambda x: None)

        s = 'UGID > 40'

        def f(obj):
            self.assertEqual(len(obj), 11)
            for ii in obj:
                item = ii.items()['UGID']
                self.assertTrue(item > 40)

        _run_(s, f)

    def test_number_in_shapefile_name(self):
        """Test number in shapefile name."""

        sc = GeomCabinet()
        path = sc.get_shp_path('state_boundaries')
        out_path = os.path.join(self.current_dir_output, '51_states.shp')
        with fiona.open(path) as source:
            with fiona.open(out_path, mode='w', driver='ESRI Shapefile', schema=source.meta['schema'],
                            crs=source.meta['crs']) as sink:
                for record in source:
                    sink.write(record)
        ret = list(GeomCabinetIterator(select_uid=[23], path=out_path))
        self.assertEqual(len(ret), 1)

    def test_iter_geoms_select_ugid_is_sorted(self):
        sc = GeomCabinet()
        with self.assertRaises(ValueError):
            list(sc.iter_geoms('state_boundaries', select_uid=[23, 18]))

    def test_iter_geoms_no_load_geoms(self):
        sc = GeomCabinet()
        it = sc.iter_geoms('state_boundaries', load_geoms=False)
        geoms = list(it)
        self.assertEqual(len(geoms), 51)
        self.assertEqual(geoms[12]['properties']['STATE_NAME'], 'New Hampshire')
        for geom in geoms:
            self.assertNotIn('geom', geom)

    def test_iter_geoms(self):
        sc = GeomCabinet()
        it = sc.iter_geoms('state_boundaries')
        geoms = list(it)
        self.assertEqual(len(geoms), 51)
        self.assertEqual(geoms[12]['properties']['STATE_NAME'], 'New Hampshire')
        for geom in geoms:
            self.assertIn(type(geom['geom']), (Polygon, MultiPolygon))

        # test with a shapefile not having a unique identifier
        env.DEFAULT_GEOM_UID = 'ggidd'
        new = self.get_shapefile_path_with_no_ugid()
        sc = GeomCabinet()
        target = list(sc.iter_geoms(path=new))
        self.assertEqual(len(target), 11)
        self.assertEqual(target[0]['properties'][env.DEFAULT_GEOM_UID], 1)
        self.assertEqual(target[3]['properties'][env.DEFAULT_GEOM_UID], 4)

        target = list(sc.iter_geoms(path=new, uid='ID'))
        self.assertNotIn(env.DEFAULT_GEOM_UID, target[9]['properties'])
        self.assertEqual(int, type(target[7]['properties']['ID']))

        target = list(sc.iter_geoms(path=new, uid='ID', as_spatial_dimension=True))
        ref = target[4]
        self.assertEqual(ref.uid[0, 0], 10)
        self.assertNotIn(env.DEFAULT_GEOM_UID, ref.properties.dtype.names)

        # test with a different geometry unique identifier
        path = self.get_shapefile_path_with_no_ugid()
        geom_select_uid = [12, 15]
        geom_uid = 'ID'
        sc = GeomCabinet()
        records = list(sc.iter_geoms(path=path, uid=geom_uid, select_uid=geom_select_uid))
        self.assertEqual(len(records), 2)
        self.assertEqual([r['properties']['ID'] for r in records], geom_select_uid)

    def test_iter_geoms_select_sql_where(self):
        sc = GeomCabinet()
        sql = 'STATE_NAME = "New Hampshire"'
        self.assertEqual(len(list(sc.iter_geoms('state_boundaries', select_sql_where=sql))), 1)

    def test_iter_geoms_select_ugid(self):
        sc = GeomCabinet()
        it = sc.iter_geoms('state_boundaries', select_uid=[13])
        geoms = list(it)
        self.assertEqual(len(geoms), 1)
        self.assertEqual(geoms[0]['properties']['STATE_NAME'], 'New Hampshire')

    def test_sql_subset(self):
        sc = GeomCabinet()
        path = sc.get_shp_path('state_boundaries')
        ds = ogr.Open(path)
        ret = ds.ExecuteSQL('select * from state_boundaries where state_name = "New Jersey"')
        ret.ResetReading()
        self.assertEqual(len(ret), 1)

    def test_get_keys(self, path=None):
        if path is not None:
            env.DIR_GEOMCABINET = path
        sc = GeomCabinet()
        ret = sc.keys()
        target_keys = ['state_boundaries']
        self.assertEqual(len(set(target_keys).intersection(set(ret))), len(target_keys))

    def test_shapefiles_not_in_folders(self):
        for dirpath, dirnames, filenames in os.walk(ocgis.env.get_geomcabinet_path()):
            for filename in filenames:
                if filename.startswith('state_boundaries') or filename.startswith('world_countries'):
                    dst = os.path.join(self.current_dir_output, filename)
                    src = os.path.join(dirpath, filename)
                    shutil.copy2(src, dst)
        self.test_get_keys(path=self.current_dir_output)
