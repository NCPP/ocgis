import re
import itertools
import os.path
from abc import ABCMeta, abstractproperty
import netCDF4 as nc
import csv
from collections import OrderedDict
from copy import deepcopy
from csv import DictReader
import tempfile
import numpy as np

from fiona.crs import from_string
from osgeo.osr import SpatialReference
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
import fiona
from shapely.geometry.geo import mapping
from shapely import wkt

import datetime
from ocgis.api.operations import OcgOperations
from ocgis.api.interpreter import OcgInterpreter
from ocgis.api.parms.definition import SpatialOperation
from ocgis.util.helpers import make_poly, project_shapely_geometry
from ocgis import exc, env, constants
from ocgis.test.base import TestBase, nc_scope
import ocgis
from ocgis.exc import ExtentError, DefinitionValidationError
from ocgis.interface.base import crs
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84, CFWGS84, WrappableCoordinateReferenceSystem
from ocgis.api.request.base import RequestDataset
from ocgis.test.test_simple.make_test_data import SimpleNcNoLevel, SimpleNc, SimpleMaskNc, \
    SimpleNc360, SimpleNcProjection, SimpleNcNoSpatialBounds, SimpleNcMultivariate
from ocgis.api.parms.definition import OutputFormat
from ocgis.interface.base.field import DerivedMultivariateField
from ocgis.util.itester import itr_products_keywords
from ocgis.util.shp_cabinet import ShpCabinetIterator
from ocgis.util.spatial.fiona_maker import FionaMaker


class TestSimpleBase(TestBase):
    __metaclass__ = ABCMeta

    base_value = None
    return_shp = False
    var = 'foo'

    @abstractproperty
    def nc_factory(self):
        pass

    @abstractproperty
    def fn(self):
        pass

    def setUp(self):
        TestBase.setUp(self)
        self.nc_factory().write()

    def get_dataset(self, time_range=None, level_range=None, time_region=None):
        uri = os.path.join(env.DIR_OUTPUT, self.fn)
        return ({'uri': uri, 'variable': self.var,
                 'time_range': time_range, 'level_range': level_range,
                 'time_region': time_region})

    def get_ops(self, kwds={}, time_range=None, level_range=None):
        dataset = self.get_dataset(time_range, level_range)
        if 'output_format' not in kwds:
            kwds.update({'output_format': 'numpy'})
        kwds.update({'dataset': dataset})
        ops = OcgOperations(**kwds)
        return (ops)

    def get_ret(self, ops=None, kwds={}, shp=False, time_range=None, level_range=None):
        """
        :param ops:
        :type ops: :class:`ocgis.api.operations.OcgOperations`
        :param dict kwds:
        :param bool shp: If ``True``, override output format to shapefile.
        :param time_range:
        :type time_range: list[:class:`datetime.datetime`]
        :param level_range:
        :type level_range: list[int]
        """

        if ops is None:
            ops = self.get_ops(kwds, time_range=time_range, level_range=level_range)
        self.ops = ops
        ret = OcgInterpreter(ops).execute()

        if shp or self.return_shp:
            kwds2 = kwds.copy()
            kwds2.update({'output_format': 'shp'})
            ops2 = OcgOperations(**kwds2)
            OcgInterpreter(ops2).execute()

        return ret

    def make_shp(self):
        ops = OcgOperations(dataset=self.dataset,
                            output_format='shp')
        OcgInterpreter(ops).execute()


class TestSimpleNoLevel(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcNoLevel
    fn = 'test_simple_spatial_no_level_01.nc'

    def test_nc_write_no_level(self):
        ret = self.get_ret(kwds={'output_format': 'nc'})
        ret2 = self.get_ret(kwds={'output_format': 'nc',
                                  'dataset': {'uri': ret, 'variable': 'foo'}, 'prefix': 'level_again'})
        self.assertNcEqual(ret, ret2, ignore_attributes={'global': ['history']})

        ds = nc.Dataset(ret)
        try:
            self.assertTrue('level' not in ds.dimensions)
            self.assertTrue('level' not in ds.variables)
        finally:
            ds.close()


class TestSimple(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    def test_meta_attrs_applied(self):
        """Test overloaded metadata attributes are applied to output calculation."""

        calc = [{'func': 'mean', 'name': 'mean', 'meta_attrs': {'this_is': 'something new', 'a_number': 5}}]
        calc_grouping = ['month']
        ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': calc_grouping, 'output_format': 'nc'})
        with nc_scope(ret) as ds:
            var = ds.variables['mean']
            self.assertEqual(var.__dict__['this_is'], 'something new')
            self.assertEqual(var.__dict__['a_number'], 5)

    def test_meta_attrs_eval_function(self):
        """Test metadata attributes applied with evaluation function."""

        calc = [{'func': 'up=foo+5', 'meta_attrs': {'this_is': 'something new', 'a_number': 5}}]
        ret = self.get_ret(kwds={'calc': calc, 'output_format': 'nc'})
        with nc_scope(ret) as ds:
            var = ds.variables['up']
            self.assertEqual(var.__dict__['this_is'], 'something new')
            self.assertEqual(var.__dict__['a_number'], 5)

    def test_selection_geometry_crs_differs(self):
        """Test selection is appropriate when CRS of selection geometry differs from source."""

        dataset = self.get_dataset()
        rd = RequestDataset(**dataset)

        ugeom = 'POLYGON((-104.000538 39.004301,-102.833871 39.215054,-102.833871 39.215054,-102.833871 39.215054,-102.879032 37.882796,-104.136022 37.867742,-104.000538 39.004301))'
        ugeom = wkt.loads(ugeom)
        from_sr = SpatialReference()
        from_sr.ImportFromEPSG(4326)
        to_sr = SpatialReference()
        to_sr.ImportFromProj4(
            '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs')
        ugeom = project_shapely_geometry(ugeom, from_sr, to_sr)
        crs = from_string(to_sr.ExportToProj4())

        # write_geom_dict({1: ugeom}, '/tmp/projected.shp', crs=crs)

        geom = [{'geom': ugeom, 'crs': crs}]
        ops = OcgOperations(dataset=rd, geom=geom)
        ret = ops.execute()

        to_test = ret[1]['foo'].variables['foo'].value[:, 0, 0, :, :]
        actual = np.loads(
            '\x80\x02cnumpy.ma.core\n_mareconstruct\nq\x01(cnumpy.ma.core\nMaskedArray\nq\x02cnumpy\nndarray\nq\x03K\x00\x85q\x04U\x01btRq\x05(K\x01K\x01K\x02K\x02\x87cnumpy\ndtype\nq\x06U\x02f8K\x00K\x01\x87Rq\x07(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U \x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@U\x04\x00\x00\x00\x00cnumpy.core.multiarray\n_reconstruct\nq\x08h\x03K\x00\x85U\x01b\x87Rq\t(K\x01)h\x07\x89U\x08@\x8c\xb5x\x1d\xaf\x15Dtbtb.')
        self.assertNumpyAll(to_test, actual)

    def test_history_attribute(self):
        ops = self.get_ops(kwds={'output_format': 'nc'})
        ret = ops.execute()
        with nc_scope(ret) as ds:
            history = ds.history
            self.assertTrue('OcgOperations' in history)
            self.assertTrue('ocgis' in history)
            self.assertTrue(len(history) > 500)

    def test_select_nearest(self):
        for spatial_operation in SpatialOperation.iter_possible():
            kwds = {'spatial_operation': spatial_operation, 'geom': [-104.0, 39.0], 'select_nearest': True}
            ops = self.get_ops(kwds=kwds)
            ret = ops.execute()
            self.assertEqual(ret[1]['foo'].variables['foo'].shape, (1, 61, 2, 1, 1))
            self.assertTrue(ret[1]['foo'].spatial.geom.point.value[0, 0].almost_equals(Point(-104.0, 39.0)))

    def test_optimizations_in_calculations(self):
        # pass optimizations to the calculation engine using operations and ensure the output values are equivalent
        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        tgd = field.temporal.get_grouping(['month'])
        optimizations = {'tgds': {rd.name: tgd}}
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            optimizations=optimizations)
        ret_with_optimizations = ops.execute()
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            optimizations=None)
        ret_without_optimizations = ops.execute()
        t1 = ret_with_optimizations[1]['foo'].variables['mean']
        t2 = ret_without_optimizations[1]['foo'].variables['mean']
        self.assertNumpyAll(t1.value, t2.value)

    def test_optimizations_in_calculations_bad_calc_grouping(self):
        # bad calculations groupings in the optimizations should be caught and raise a value error
        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        tgd1 = field.temporal.get_grouping('all')
        tgd2 = field.temporal.get_grouping(['year', 'month'])
        tgd3 = field.temporal.get_grouping([[3]])
        for t in [tgd1, tgd2, tgd3]:
            optimizations = {'tgds': {rd.alias: t}}
            calc = [{'func': 'mean', 'name': 'mean'}]
            calc_grouping = ['month']
            ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                                optimizations=optimizations)
            with self.assertRaises(ValueError):
                ops.execute()

    def test_operations_abstraction_used_for_subsetting(self):
        ret = self.get_ret(kwds={'abstraction': 'point'})
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.abstraction, 'point')
        self.assertIsInstance(ref.spatial.abstraction_geometry.value[0, 0], Point)
        with self.assertRaises(ValueError):
            ref.get_intersects(Point(-103., 38.))
        sub = ref.get_intersects(Point(-103., 38.).buffer(0.75), use_spatial_index=env.USE_SPATIAL_INDEX)
        self.assertEqual(sub.shape, (1, 61, 2, 1, 1))

    def test_to_csv_shp_and_shape_with_point_subset(self):
        rd = RequestDataset(**self.get_dataset())
        geom = [-103.5, 38.5]
        for of in [constants.OUTPUT_FORMAT_CSV_SHAPEFILE, constants.OUTPUT_FORMAT_SHAPEFILE]:
            ops = ocgis.OcgOperations(dataset=rd, geom=geom, output_format=of, prefix=of)
            ops.execute()

    def test_overwrite_add_auxiliary_files(self):
        # if overwrite is true, we should be able to write the nc output multiple times
        env.OVERWRITE = True
        kwds = {'output_format': constants.OUTPUT_FORMAT_NETCDF, 'add_auxiliary_files': False}
        self.get_ret(kwds=kwds)
        self.get_ret(kwds=kwds)
        # switching the argument to false will result in an IOError
        env.OVERWRITE = False
        with self.assertRaises(IOError):
            self.get_ret(kwds=kwds)

    def test_units_calendar_on_time_bounds(self):
        """Test units and calendar are copied to the time bounds."""

        rd = self.get_dataset()
        ops = ocgis.OcgOperations(dataset=rd, output_format=constants.OUTPUT_FORMAT_NETCDF)
        ret = ops.execute()
        with nc_scope(ret) as ds:
            time_attrs = deepcopy(ds.variables['time'].__dict__)
            time_attrs.pop('bounds')
            time_attrs.pop('axis')
            self.assertEqual(dict(time_attrs), dict(ds.variables['time_bnds'].__dict__))

    def test_units_calendar_on_time_bounds_calculation(self):
        rd = self.get_dataset()
        ops = ocgis.OcgOperations(dataset=rd, output_format='nc', calc=[{'func': 'mean', 'name': 'my_mean'}],
                                  calc_grouping=['month'])
        ret = ops.execute()
        with nc_scope(ret) as ds:
            time = ds.variables['time']
            bound = ds.variables['climatology_bounds']
            self.assertEqual(time.units, bound.units)
            self.assertEqual(bound.calendar, bound.calendar)

    def test_add_auxiliary_files_false_csv_nc(self):
        rd = self.get_dataset()
        for output_format in ['csv', 'nc']:
            dir_output = tempfile.mkdtemp(dir=self.current_dir_output)
            ops = ocgis.OcgOperations(dataset=rd, output_format=output_format, add_auxiliary_files=False,
                                      dir_output=dir_output)
            ret = ops.execute()
            filename = 'ocgis_output.{0}'.format(output_format)
            self.assertEqual(os.listdir(ops.dir_output), [filename])
            self.assertEqual(ret, os.path.join(dir_output, filename))

            # attempting the operation again will work if overwrite is True
            ocgis.env.OVERWRITE = True
            ops = ocgis.OcgOperations(dataset=rd, output_format=output_format, add_auxiliary_files=False,
                                      dir_output=dir_output)
            ops.execute()

    def test_multiple_request_datasets(self):
        aliases = ['foo1', 'foo2', 'foo3', 'foo4']
        rds = []
        for alias in aliases:
            rd = self.get_dataset()
            rd['alias'] = alias
            rds.append(rd)
        ops = ocgis.OcgOperations(dataset=rds, output_format='csv')
        ret = ops.execute()
        with open(ret, 'r') as f:
            reader = csv.DictReader(f)
            test_aliases = set([row['ALIAS'] for row in reader])
        self.assertEqual(set(aliases), test_aliases)

    def test_agg_selection(self):
        features = [
            {'NAME': 'a',
             'wkt': 'POLYGON((-105.020430 40.073118,-105.810753 39.327957,-105.660215 38.831183,-104.907527 38.763441,-104.004301 38.816129,-103.643011 39.802151,-103.643011 39.802151,-103.643011 39.802151,-103.643011 39.802151,-103.959140 40.118280,-103.959140 40.118280,-103.959140 40.118280,-103.959140 40.118280,-104.327957 40.201075,-104.327957 40.201075,-105.020430 40.073118))'},
            {'NAME': 'b',
             'wkt': 'POLYGON((-102.212903 39.004301,-102.905376 38.906452,-103.311828 37.694624,-103.326882 37.295699,-103.898925 37.220430,-103.846237 36.746237,-102.619355 37.107527,-102.634409 37.724731,-101.874194 37.882796,-102.212903 39.004301))'},
            {'NAME': 'c',
             'wkt': 'POLYGON((-105.336559 37.175269,-104.945161 37.303226,-104.726882 37.175269,-104.696774 36.844086,-105.043011 36.693548,-105.283871 36.640860,-105.336559 37.175269))'},
            {'NAME': 'd',
             'wkt': 'POLYGON((-102.318280 39.741935,-103.650538 39.779570,-103.620430 39.448387,-103.349462 39.433333,-103.078495 39.606452,-102.325806 39.613978,-102.325806 39.613978,-102.333333 39.741935,-102.318280 39.741935))'},
        ]

        geom = []
        for feature in features:
            geom.append({'geom': wkt.loads(feature['wkt']), 'properties': {'NAME': feature['NAME']}})
        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, output_format='shp', agg_selection=True)
        ret = ops.execute()
        ugid_path = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')
        with fiona.open(ugid_path) as f:
            self.assertEqual(len(f), 1)

        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, agg_selection=True)
        ret = ops.execute()
        self.assertEqual(ret[1]['foo'].spatial.shape, (4, 4))
        self.assertEqual(len(ret), 1)
        self.assertEqual(len(ret.geoms), 1)

        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, output_format='nc',
                            prefix='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            ref = ds.variables['foo']
            self.assertFalse(ref[:].mask.all())

        ops = OcgOperations(dataset=self.get_dataset(), agg_selection=True)
        ops.execute()

    def test_point_subset(self):
        ops = self.get_ops(kwds={'geom': [-103.5, 38.5, ]})
        self.assertEqual(type(ops.geom[0].geom.point.value[0, 0]), Point)
        ret = ops.execute()
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.grid.shape, (4, 4))

        ops = self.get_ops(kwds={'geom': [-103, 38, ], 'search_radius_mult': 0.01})
        ret = ops.execute()
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.grid.shape, (1, 1))
        self.assertTrue(ref.spatial.geom.polygon.value[0, 0].intersects(ops.geom[0].geom.point.value[0, 0]))

        ops = self.get_ops(kwds={'geom': [-103, 38, ], 'abstraction': 'point', 'search_radius_mult': 0.01})
        ret = ops.execute()
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.grid.shape, (1, 1))
        # this is a point abstraction. polygons are not available.
        self.assertIsNone(ref.spatial.geom.polygon)
        # self.assertTrue(ref.spatial.geom.polygon.value[0,0].intersects(ops.geom[0].geom.point.value[0, 0]))

    def test_slicing(self):
        ops = self.get_ops(kwds={'slice': [None, None, 0, [0, 2], [0, 2]]})
        ret = ops.execute()
        ref = ret.gvu(1, 'foo')
        self.assertTrue(np.all(ref.flatten() == 1.0))
        self.assertEqual(ref.shape, (1, 61, 1, 2, 2))

        ops = self.get_ops(kwds={'slice': [0, None, None, [1, 3], [1, 3]]})
        ret = ops.execute()
        ref = ret.gvu(1, 'foo').data
        self.assertTrue(np.all(np.array([1., 2., 3., 4.] == ref[0, 0, 0, :].flatten())))

        # pass only three slices
        with self.assertRaises(DefinitionValidationError):
            self.get_ops(kwds={'slice': [None, [1, 3], [1, 3]]})

    def test_file_only(self):
        ret = self.get_ret(
            kwds={'output_format': 'nc', 'file_only': True, 'calc': [{'func': 'mean', 'name': 'my_mean'}],
                  'calc_grouping': ['month']})
        try:
            ds = nc.Dataset(ret, 'r')
            self.assertTrue(isinstance(ds.variables['my_mean'][:].sum(), np.ma.core.MaskedConstant))
            self.assertEqual(set(ds.variables['my_mean'].ncattrs()),
                             set([u'_FillValue', u'units', u'long_name', u'standard_name', 'grid_mapping']))
        finally:
            ds.close()

        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'file_only': True, 'output_format': 'shp'})

        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'file_only': True})

    def test_return_all(self):
        ret = self.get_ret()

        # confirm size of geometry array
        ref = ret[1][self.var].spatial
        shps = [ref.geom, ref.grid, ref.geom.uid, ref.grid.uid]
        for attr in shps:
            self.assertEqual(attr.shape, (4, 4))

        # confirm value array
        ref = ret.gvu(1, self.var)
        self.assertEqual(ref.shape, (1, 61, 2, 4, 4))
        for tidx, lidx in itertools.product(range(0, 61), range(0, 2)):
            slice = ref[0, tidx, lidx, :, :]
            idx = self.base_value == slice
            self.assertTrue(np.all(idx))

    def test_aggregate(self):
        ret = self.get_ret(kwds={'aggregate': True})

        # test area-weighting
        ref = ret.gvu(1, self.var)
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))

        # test geometry reduction
        ref = ret[1][self.var]
        self.assertEqual(ref.spatial.shape, (1, 1))

    def test_time_level_subset(self):
        ret = self.get_ret(time_range=[datetime.datetime(2000, 3, 1),
                                       datetime.datetime(2000, 3, 31, 23)],
                           level_range=[1, 1])
        ref = ret.gvu(1, self.var)
        self.assertEqual(ref.shape, (1, 31, 1, 4, 4))

    def test_time_level_subset_aggregate(self):
        ret = self.get_ret(kwds={'aggregate': True},
                           time_range=[datetime.datetime(2000, 3, 1), datetime.datetime(2000, 3, 31)],
                           level_range=[1, 1], )
        ref = ret.gvu(1, self.var)
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))
        ref = ret[1][self.var]
        self.assertEqual(ref.level.value.shape, (1,))

    def test_time_region_subset(self):
        """Test subsetting a Field object by a time region."""

        rd = ocgis.RequestDataset(uri=os.path.join(env.DIR_OUTPUT, self.fn),
                                  variable=self.var)
        ops = ocgis.OcgOperations(dataset=rd)
        ret = ops.execute()
        all = ret[1]['foo'].temporal.value_datetime

        def get_ref(month, year):
            rd = ocgis.RequestDataset(uri=os.path.join(env.DIR_OUTPUT, self.fn),
                                      variable=self.var,
                                      time_region={'month': month, 'year': year})
            ops = ocgis.OcgOperations(dataset=rd)
            ret = ops.execute()
            ref = ret[1]['foo'].temporal.value_datetime
            months = set([i.month for i in ref.flat])
            years = set([i.year for i in ref.flat])
            if month is not None:
                for m in month:
                    self.assertTrue(m in months)
            if year is not None:
                for m in year:
                    self.assertTrue(m in years)
            return (ref)

        ref = get_ref(None, None)
        self.assertTrue(np.all(ref == all))

        ref = get_ref([3], None)
        self.assertEqual(ref.shape[0], 31)

        ref = get_ref([3, 4], None)
        self.assertTrue(np.all(ref == all))

        ref = get_ref([4], None)
        self.assertEqual(ref.shape[0], 30)

        ref = get_ref(None, [2000])
        self.assertTrue(np.all(ref == all))

        with self.assertRaises(ExtentError):
            get_ref([1], None)

    def test_spatial_aggregate_arbitrary(self):
        poly = Polygon(((-103.5, 39.5), (-102.5, 38.5), (-103.5, 37.5), (-104.5, 38.5)))
        ret2 = self.get_ret(kwds={'output_format': 'numpy', 'geom': poly,
                                  'prefix': 'subset', 'spatial_operation': 'clip', 'aggregate': True})
        self.assertEqual(ret2.gvu(1, self.var).data.mean(), 2.5)

    def test_spatial(self):
        # intersects
        geom = make_poly((37.5, 39.5), (-104.5, -102.5))
        ret = self.get_ret(kwds={'geom': geom})
        ref = ret[1][self.var]
        gids = set([6, 7, 10, 11])
        ret_gids = set(ref.spatial.uid.compressed())
        self.assertEqual(ret_gids, gids)
        to_test = ref.variables[self.var].value[0, 0, 0, :, :]
        self.assertEqual(to_test.shape, (2, 2))
        self.assertTrue(np.all(to_test == np.array([[1.0, 2.0], [3.0, 4.0]])))

        # intersection
        geom = make_poly((38, 39), (-104, -103))
        ret = self.get_ret(kwds={'geom': geom, 'spatial_operation': 'clip'})
        self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), 4)
        self.assertEqual(ret[1][self.var].variables[self.var].value.shape, (1, 61, 2, 2, 2))
        ref = ret[1][self.var].variables[self.var].value
        self.assertTrue(np.all(ref[0, 0, :, :] == np.array([[1, 2], [3, 4]], dtype=float)))
        # # compare areas to intersects returns
        ref = ret[1][self.var]
        intersection_areas = [g.area for g in ref.spatial.abstraction_geometry.value.flat]
        for ii in intersection_areas:
            self.assertAlmostEqual(ii, 0.25)

        # intersection + aggregation
        geom = make_poly((38, 39), (-104, -103))
        ret = self.get_ret(kwds={'geom': geom, 'spatial_operation': 'clip', 'aggregate': True})
        ref = ret[1][self.var]
        self.assertEqual(len(ref.spatial.uid.flatten()), 1)
        self.assertEqual(ref.spatial.abstraction_geometry.value.flatten()[0].area, 1.0)
        self.assertEqual(ref.variables[self.var].value.flatten().mean(), 2.5)

    def test_empty_intersection(self):
        geom = make_poly((20, 25), (-90, -80))

        with self.assertRaises(exc.ExtentError):
            self.get_ret(kwds={'geom': geom})

        ret = self.get_ret(kwds={'geom': geom, 'allow_empty': True})
        self.assertEqual(ret[1]['foo'], None)

    def test_empty_time_subset(self):
        ds = self.get_dataset(time_range=[datetime.datetime(2900, 1, 1), datetime.datetime(3100, 1, 1)])

        ops = OcgOperations(dataset=ds)
        with self.assertRaises(ExtentError):
            ops.execute()

        ops = OcgOperations(dataset=ds, allow_empty=True)
        ret = ops.execute()
        self.assertEqual(ret[1]['foo'], None)

    def test_snippet(self):
        ret = self.get_ret(kwds={'snippet': True})
        ref = ret.gvu(1, self.var)
        self.assertEqual(ref.shape, (1, 1, 1, 4, 4))
        with nc_scope(os.path.join(self.current_dir_output, self.fn)) as ds:
            to_test = ds.variables['foo'][0, 0, :, :].reshape(1, 1, 1, 4, 4)
            self.assertNumpyAll(to_test, ref.data)

        calc = [{'func': 'mean', 'name': 'my_mean'}]
        group = ['month', 'year']
        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'calc': calc, 'calc_grouping': group, 'snippet': True})

    def test_snippet_time_region(self):
        with self.assertRaises(DefinitionValidationError):
            rd = self.get_dataset(time_region={'month': [1]})
            OcgOperations(dataset=rd, snippet=True)

    def test_calc(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        group = ['month', 'year']

        # raw
        ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': group})
        ref = ret.gvu(1, 'my_mean')
        self.assertEqual(ref.shape, (1, 2, 2, 4, 4))
        with self.assertRaises(KeyError):
            ret.gvu(1, 'n')

        # aggregated
        for calc_raw in [True, False]:
            ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': group, 'aggregate': True, 'calc_raw': calc_raw})
            ref = ret.gvu(1, 'my_mean')
            self.assertEqual(ref.shape, (1, 2, 2, 1, 1))
            self.assertEqual(ref.flatten().mean(), 2.5)
            self.assertDictEqual(ret[1]['foo'].variables['my_mean'].attrs,
                                 {'long_name': 'Mean', 'standard_name': 'mean'})

    def test_calc_multivariate(self):
        rd1 = self.get_dataset()
        rd1['alias'] = 'var1'
        rd2 = self.get_dataset()
        rd2['alias'] = 'var2'
        calc = [{'name': 'divide', 'func': 'divide', 'kwds': {'arr1': 'var1', 'arr2': 'var2'}}]

        calc_grouping = [None, ['month']]
        for cg in calc_grouping:
            ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=cg)
            ret = ops.execute()
            ref = ret.gvu(1, 'divide').shape
            if cg is None:
                self.assertEqual(ref, (1, 61, 2, 4, 4))
            else:
                self.assertEqual(ref, (1, 2, 2, 4, 4))

    def test_calc_eval(self):
        rd = self.get_dataset()
        calc = 'foo2=foo+4'
        ocgis.env.OVERWRITE = True
        for of in OutputFormat.iter_possible():
            ops = ocgis.OcgOperations(dataset=rd, calc=calc, output_format=of)
            ret = ops.execute()
            if of == 'nc':
                with nc_scope(ret) as ds:
                    self.assertEqual(ds.variables['foo2'][:].mean(), 6.5)

    def test_calc_eval_multivariate(self):
        rd = self.get_dataset()
        rd2 = self.get_dataset()
        rd2['alias'] = 'foo2'
        calc = 'foo3=foo+foo2+4'
        ocgis.env.OVERWRITE = True
        for of in OutputFormat.iter_possible():
            try:
                ops = ocgis.OcgOperations(dataset=[rd, rd2], calc=calc, output_format=of,
                                          slice=[None, [0, 10], None, None, None])
            except DefinitionValidationError:
                self.assertEqual(of, 'esmpy')
                continue
            ret = ops.execute()
            if of == 'numpy':
                self.assertIsInstance(ret[1]['foo_foo2'], DerivedMultivariateField)
            if of == 'nc':
                with nc_scope(ret) as ds:
                    self.assertEqual(ds.variables['foo3'][:].mean(), 9.0)

    def test_nc_conversion(self):
        rd = self.get_dataset()
        ops = OcgOperations(dataset=rd, output_format='nc')
        ret = self.get_ret(ops)

        self.assertNcEqual(ret, rd['uri'], ignore_attributes={'global': ['history'],
                                                              'time_bnds': ['calendar', 'units'],
                                                              rd['variable']: ['grid_mapping'],
                                                              'time': ['axis'],
                                                              'level': ['axis'],
                                                              'latitude': ['axis'],
                                                              'longitude': ['axis']},
                           ignore_variables=['latitude_longitude'])

        with self.nc_scope(ret) as ds:
            expected = {'time': 'T', 'level': 'Z', 'latitude': 'Y', 'longitude': 'X'}
            for k, v in expected.iteritems():
                var = ds.variables[k]
                self.assertEqual(var.axis, v)
        with self.nc_scope(ret) as ds:
            self.assertEqual(ds.file_format, constants.NETCDF_DEFAULT_DATA_MODEL)

    def test_nc_conversion_calc(self):
        calc_grouping = ['month']
        calc = [{'func': 'mean', 'name': 'my_mean'},
                {'func': 'std', 'name': 'my_stdev'}]
        kwds = dict(calc_grouping=calc_grouping, calc=calc, output_format='nc')
        ret = self.get_ret(kwds=kwds)

        ds = nc.Dataset(ret)
        try:
            for alias in ['my_mean', 'my_stdev']:
                self.assertEqual(ds.variables[alias].shape, (2, 2, 4, 4))

            # output variable should not have climatology bounds and no time bounds directly
            with self.assertRaises(AttributeError):
                ds.variables['time'].bounds

            self.assertEqual(ds.variables['time'].climatology, 'climatology_bounds')
            self.assertEqual(ds.variables['climatology_bounds'].shape, (2, 2))
        finally:
            ds.close()

    def test_nc_conversion_multiple_request_datasets(self):
        rd1 = self.get_dataset()
        rd2 = self.get_dataset()
        rd2['alias'] = 'foo2'
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=[rd1, rd2], output_format='nc')

    def test_nc_conversion_level_subset(self):
        rd = self.get_dataset(level_range=[1, 1])
        ops = OcgOperations(dataset=rd, output_format='nc', prefix='no_level')
        no_level = ops.execute()

        ops = OcgOperations(dataset={'uri': no_level, 'variable': 'foo'}, output_format='nc', prefix='no_level_again')
        no_level_again = ops.execute()

        self.assertNcEqual(no_level, no_level_again, ignore_attributes={'global': ['history']})

        ds = nc.Dataset(no_level_again)
        try:
            ref = ds.variables['foo'][:]
            self.assertEqual(ref.shape[1], 1)
        finally:
            ds.close()

    def test_nc_conversion_multivariate_calculation(self):
        rd1 = self.get_dataset()
        rd2 = self.get_dataset()
        rd2['alias'] = 'foo2'
        calc = [{'func': 'divide', 'name': 'my_divide', 'kwds': {'arr1': 'foo', 'arr2': 'foo2'}}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=calc_grouping,
                            output_format='nc')
        ret = ops.execute()

        ds = nc.Dataset(ret)
        try:
            ref = ds.variables['my_divide'][:]
            self.assertEqual(ref.shape, (2, 2, 4, 4))
            self.assertEqual(np.unique(ref)[0], 1.)
        finally:
            ds.close()

    def test_shp_projection(self):
        output_crs = CoordinateReferenceSystem(epsg=2163)
        ret = self.get_ret(kwds=dict(output_crs=output_crs, output_format='shp'))
        with fiona.open(ret) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), output_crs)

    def test_geojson_projection(self):
        output_crs = CoordinateReferenceSystem(epsg=2163)
        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds=dict(output_crs=output_crs, output_format='geojson'))

    def test_limiting_headers(self):
        headers = ['value']
        ops = OcgOperations(dataset=self.get_dataset(), headers=headers, output_format='csv')
        ret = ops.execute()
        with open(ret) as f:
            reader = DictReader(f)
            self.assertEqual(reader.fieldnames, [c.upper() for c in constants.HEADERS_REQUIRED] + ['VALUE'])

        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=self.get_dataset(), headers=['foo'], output_format='csv')

    def test_empty_dataset_for_operations(self):
        with self.assertRaises(DefinitionValidationError):
            OcgOperations()

    def test_shp_conversion(self):
        ocgis.env.OVERWRITE = True
        calc = [None, [{'func': 'mean', 'name': 'my_mean'}]]
        group = ['month', 'year']
        for c in calc:
            ops = OcgOperations(dataset=self.get_dataset(),
                                output_format='shp',
                                calc_grouping=group,
                                calc=c)
            ret = self.get_ret(ops)

            if c is None:
                with fiona.open(ret) as f:
                    schema_properties = OrderedDict(
                        [(u'DID', 'int:10'), (u'VID', 'int:10'), (u'UGID', 'int:10'), (u'TID', 'int:10'),
                         (u'LID', 'int:10'), (u'GID', 'int:10'), (u'VARIABLE', 'str:80'), (u'ALIAS', 'str:80'),
                         (u'TIME', 'str:80'), (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'),
                         (u'LEVEL', 'int:10'), (u'VALUE', 'float:24.15')])
                    self.assertDictEqual(f.meta['schema']['properties'], schema_properties)
                    self.assertDictEqual(f.meta, {'crs': {'init': u'epsg:4326'},
                                                  'driver': u'ESRI Shapefile',
                                                  'schema': {'geometry': 'Polygon', 'properties': schema_properties}})
                    self.assertEqual(len(f), 1952)
                    record_properties = OrderedDict(
                        [(u'DID', 1), (u'VID', 1), (u'UGID', 1), (u'TID', 11), (u'LID', 2), (u'GID', 5.0),
                         (u'VARIABLE', u'foo'), (u'ALIAS', u'foo'), (u'TIME', '2000-03-11 12:00:00'), (u'YEAR', 2000),
                         (u'MONTH', 3), (u'DAY', 11), (u'LEVEL', 150), (u'VALUE', 1.0)])
                    record = list(f)[340]
                    self.assertDictEqual(record['properties'], record_properties)
                    record_coordinates = [
                        [(-105.5, 37.5), (-105.5, 38.5), (-104.5, 38.5), (-104.5, 37.5), (-105.5, 37.5)]]
                    self.assertDictEqual(record, {'geometry': {'type': 'Polygon',
                                                               'coordinates': record_coordinates},
                                                  'type': 'Feature',
                                                  'id': '340',
                                                  'properties': record_properties})
            else:
                with fiona.open(ret) as f:
                    self.assertDictEqual(f.meta, {'crs': {'init': u'epsg:4326'}, 'driver': u'ESRI Shapefile',
                                                  'schema': {'geometry': 'Polygon', 'properties': OrderedDict(
                                                      [(u'DID', 'int:10'), (u'VID', 'int:10'), (u'CID', 'int:10'),
                                                       (u'UGID', 'int:10'), (u'TID', 'int:10'), (u'LID', 'int:10'),
                                                       (u'GID', 'int:10'), (u'VARIABLE', 'str:80'),
                                                       (u'ALIAS', 'str:80'), (u'CALC_KEY', 'str:80'),
                                                       (u'CALC_ALIAS', 'str:80'), (u'TIME', 'str:80'),
                                                       (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'),
                                                       (u'LEVEL', 'int:10'), (u'VALUE', 'float:24.15')])}})
                    self.assertEqual(len(f), 64)

    def test_shp_conversion_with_external_geometries(self):

        def _make_record_(wkt_str, ugid, state_name):
            geom = wkt.loads(wkt_str)
            record = {'geometry': mapping(geom),
                      'properties': {'UGID': ugid, 'STATE_NAME': state_name}}
            return (record)

        nebraska = 'POLYGON((-101.407393 40.001003,-102.051535 39.998918,-102.047545 40.342644,-102.047620 40.431077,-102.046031 40.697319,-102.046992 40.743130,-102.047739 40.998071,-102.621257 41.000214,-102.652271 40.998124,-103.382956 41.000316,-103.572316 40.999648,-104.051705 41.003211,-104.054012 41.388085,-104.055500 41.564222,-104.053615 41.698218,-104.053513 41.999815,-104.056219 42.614669,-104.056199 43.003062,-103.501464 42.998618,-103.005875 42.999354,-102.788384 42.995303,-102.086701 42.989887,-101.231737 42.986843,-100.198142 42.991095,-99.532790 42.992335,-99.253971 42.992389,-98.497651 42.991778,-98.457444 42.937160,-98.391204 42.920135,-98.310339 42.881794,-98.167826 42.839571,-98.144869 42.835794,-98.123117 42.820223,-98.121820 42.808360,-98.033140 42.769192,-97.995144 42.766812,-97.963558 42.773690,-97.929477 42.792324,-97.889941 42.831271,-97.888659 42.855807,-97.818643 42.866587,-97.797028 42.849597,-97.772186 42.846164,-97.725250 42.858008,-97.685752 42.836837,-97.634970 42.861285,-97.570654 42.847990,-97.506132 42.860136,-97.483159 42.857157,-97.457263 42.850443,-97.389306 42.867433,-97.311414 42.861771,-97.271457 42.850014,-97.243189 42.851826,-97.224443 42.841202,-97.211831 42.812573,-97.161422 42.798619,-97.130469 42.773923,-97.015139 42.759542,-96.979593 42.758313,-96.970003 42.752065,-96.977869 42.727308,-96.970773 42.721147,-96.908234 42.731699,-96.810140 42.704084,-96.810437 42.681341,-96.799344 42.670019,-96.722658 42.668592,-96.699060 42.657715,-96.694596 42.641163,-96.715273 42.621907,-96.714059 42.612302,-96.636672 42.550731,-96.629294 42.522693,-96.605467 42.507236,-96.584753 42.518287,-96.547215 42.520499,-96.494701 42.488459,-96.439394 42.489240,-96.396074 42.467401,-96.397890 42.441793,-96.417628 42.414777,-96.411761 42.380918,-96.424175 42.349279,-96.389781 42.328789,-96.368700 42.298023,-96.342881 42.282081,-96.332658 42.260307,-96.337708 42.229522,-96.363512 42.214042,-96.352165 42.168185,-96.285123 42.123452,-96.265483 42.048897,-96.238725 42.028438,-96.236093 42.001258,-96.202842 41.996615,-96.185217 41.980685,-96.147328 41.966254,-96.145870 41.924907,-96.159970 41.904151,-96.135623 41.862620,-96.076417 41.791469,-96.099321 41.752975,-96.099771 41.731563,-96.085557 41.704987,-96.122202 41.694913,-96.120264 41.684094,-96.099306 41.654680,-96.111307 41.599006,-96.080835 41.576000,-96.091936 41.563145,-96.085840 41.537522,-96.050172 41.524335,-96.004592 41.536663,-95.993965 41.528103,-95.996688 41.511517,-96.013451 41.492994,-96.006897 41.481954,-95.953185 41.472387,-95.935065 41.462381,-95.940056 41.394805,-95.942895 41.340077,-95.889107 41.301389,-95.897591 41.286863,-95.911202 41.308469,-95.930230 41.302056,-95.910981 41.225245,-95.922250 41.207854,-95.916100 41.194063,-95.859198 41.180537,-95.859801 41.166865,-95.876685 41.164202,-95.858274 41.109187,-95.878804 41.065871,-95.859539 41.035002,-95.860897 41.002650,-95.837603 40.974258,-95.836541 40.901108,-95.834396 40.870300,-95.846435 40.848332,-95.851790 40.792600,-95.876616 40.730436,-95.767999 40.643117,-95.757546 40.620904,-95.767479 40.589048,-95.763412 40.549707,-95.737036 40.532373,-95.692066 40.524129,-95.687413 40.561170,-95.675693 40.565835,-95.662944 40.558729,-95.658060 40.530332,-95.684970 40.512205,-95.695361 40.485338,-95.636817 40.396390,-95.634185 40.358800,-95.616201 40.346497,-95.617933 40.331418,-95.645553 40.322346,-95.646827 40.309109,-95.595532 40.309776,-95.547137 40.266215,-95.476822 40.226855,-95.466636 40.213255,-95.460952 40.173995,-95.422476 40.131743,-95.392813 40.115416,-95.384542 40.095362,-95.403784 40.080379,-95.413764 40.048111,-95.390532 40.043750,-95.371244 40.028751,-95.345067 40.024974,-95.308697 39.999407,-95.329701 39.992595,-95.780700 39.993489,-96.001253 39.995159,-96.240598 39.994503,-96.454038 39.994172,-96.801420 39.994476,-96.908287 39.996154,-97.361912 39.997380,-97.816589 39.999729,-97.929588 39.998452,-98.264165 39.998434,-98.504479 39.997129,-98.720632 39.998461,-99.064747 39.998338,-99.178201 39.999577,-99.627859 40.002987,-100.180910 40.000478,-100.191111 40.000585,-100.735049 39.999172,-100.754856 40.000198,-101.322148 40.001821,-101.407393 40.001003))'
        kansas = 'POLYGON((-95.071931 37.001478,-95.406622 37.000615,-95.526019 37.001018,-95.785748 36.998114,-95.957961 37.000083,-96.006049 36.998333,-96.519187 37.000577,-96.748696 37.000166,-97.137693 36.999808,-97.465405 36.996467,-97.804250 36.998567,-98.104529 36.998671,-98.347143 36.999061,-98.540219 36.998376,-98.999516 36.998072,-99.437473 36.994558,-99.544639 36.995463,-99.999261 36.995417,-100.088574 36.997652,-100.634245 36.997832,-100.950587 36.996661,-101.071604 36.997466,-101.553676 36.996693,-102.024519 36.988875,-102.037207 36.988994,-102.042010 37.386279,-102.044456 37.641474,-102.043976 37.734398,-102.046061 38.253822,-102.045549 38.263343,-102.047584 38.615499,-102.047568 38.692550,-102.048972 39.037003,-102.047874 39.126753,-102.048801 39.562803,-102.049442 39.568693,-102.051535 39.998918,-101.407393 40.001003,-101.322148 40.001821,-100.754856 40.000198,-100.735049 39.999172,-100.191111 40.000585,-100.180910 40.000478,-99.627859 40.002987,-99.178201 39.999577,-99.064747 39.998338,-98.720632 39.998461,-98.504479 39.997129,-98.264165 39.998434,-97.929588 39.998452,-97.816589 39.999729,-97.361912 39.997380,-96.908287 39.996154,-96.801420 39.994476,-96.454038 39.994172,-96.240598 39.994503,-96.001253 39.995159,-95.780700 39.993489,-95.329701 39.992595,-95.308697 39.999407,-95.240961 39.942105,-95.207597 39.938176,-95.193963 39.910180,-95.150551 39.908054,-95.100722 39.869865,-95.063246 39.866538,-95.033506 39.877844,-95.021772 39.896978,-94.965023 39.900823,-94.938243 39.896081,-94.936511 39.849386,-94.923876 39.833131,-94.898324 39.828332,-94.888505 39.817400,-94.899323 39.793775,-94.933267 39.782773,-94.935114 39.775426,-94.921800 39.757841,-94.877067 39.760679,-94.871185 39.754118,-94.877860 39.739305,-94.905678 39.726755,-94.930856 39.727026,-94.953142 39.736501,-94.961786 39.732038,-94.978570 39.684988,-95.028292 39.661913,-95.056017 39.625689,-95.053613 39.586776,-95.108988 39.560692,-95.102037 39.532848,-95.047599 39.485328,-95.040511 39.462940,-94.986204 39.439461,-94.958494 39.411447,-94.925748 39.381266,-94.898281 39.380640,-94.911343 39.340121,-94.907681 39.323028,-94.881107 39.286046,-94.833476 39.261766,-94.820819 39.211004,-94.790049 39.196883,-94.730531 39.171256,-94.675514 39.174922,-94.646407 39.158427,-94.612653 39.151649,-94.601224 39.141227,-94.608137 39.112801,-94.609281 39.044667,-94.612469 38.837109,-94.613148 38.737222,-94.618717 38.471473,-94.619053 38.392032,-94.617330 38.055784,-94.616735 38.030387,-94.619293 37.679869,-94.618996 37.650374,-94.618764 37.360766,-94.618977 37.327732,-94.620664 37.060147,-94.620379 36.997046,-95.032745 37.000779,-95.071931 37.001478))'
        fiona_crs = crs.WGS84().value
        fiona_properties = {'UGID': 'int', 'STATE_NAME': 'str'}
        fiona_path = os.path.join(self.current_dir_output, 'states.shp')
        fiona_schema = {'geometry': 'Polygon',
                        'properties': fiona_properties}
        with fiona.open(fiona_path, 'w', driver='ESRI Shapefile', crs=fiona_crs, schema=fiona_schema) as f:
            record_nebraska = _make_record_(nebraska, 1, 'Nebraska')
            record_kansas = _make_record_(kansas, 2, 'Kansas')
            f.write(record_nebraska)
            f.write(record_kansas)

        ocgis.env.DIR_SHPCABINET = self.current_dir_output
        ops = OcgOperations(dataset=self.get_dataset(),
                            geom='states',
                            output_format='shp')
        ret = ops.execute()

        output_folder = os.path.join(self.current_dir_output, ops.prefix)
        contents = os.listdir(output_folder)
        self.assertEqual(set(contents),
                         set(['ocgis_output_metadata.txt', 'ocgis_output_source_metadata.txt', 'ocgis_output_ugid.shp',
                              'ocgis_output_ugid.dbf', 'ocgis_output_ugid.cpg', 'ocgis_output.dbf', 'ocgis_output.log',
                              'ocgis_output.shx', 'ocgis_output.shp', 'ocgis_output_ugid.shx', 'ocgis_output.cpg',
                              'ocgis_output.prj', 'ocgis_output_ugid.prj', 'ocgis_output_did.csv']))

        with fiona.open(ret) as f:
            rows = list(f)
            fiona_meta = deepcopy(f.meta)
            fiona_crs = fiona_meta.pop('crs')
            self.assertEqual(CoordinateReferenceSystem(value=fiona_crs), WGS84())
            properties = OrderedDict(
                [(u'DID', 'int:10'), (u'VID', 'int:10'), (u'UGID', 'int:10'), (u'TID', 'int:10'), (u'LID', 'int:10'),
                 (u'GID', 'int:10'), (u'VARIABLE', 'str:80'), (u'ALIAS', 'str:80'), (u'TIME', 'str:80'),
                 (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'), (u'LEVEL', 'int:10'),
                 (u'VALUE', 'float:24.15')])
            self.assertDictEqual(fiona_meta['schema']['properties'], properties)
            self.assertDictEqual(fiona_meta, {'driver': u'ESRI Shapefile',
                                              'schema': {'geometry': 'Polygon',
                                                         'properties': properties}})

        self.assertEqual(len(rows), 610)
        ugids = set([r['properties']['UGID'] for r in rows])
        self.assertEqual(ugids, set([1, 2]))
        properties = OrderedDict(
            [(u'DID', 1), (u'VID', 1), (u'UGID', 2), (u'TID', 26), (u'LID', 1), (u'GID', 16.0), (u'VARIABLE', u'foo'),
             (u'ALIAS', u'foo'), (u'TIME', '2000-03-26 12:00:00'), (u'YEAR', 2000), (u'MONTH', 3), (u'DAY', 26),
             (u'LEVEL', 50),
             (u'VALUE', 4.0)])
        self.assertDictEqual(properties, rows[325]['properties'])
        self.assertEqual(rows[325], {'geometry': {'type': 'Polygon', 'coordinates': [
            [(-102.5, 39.5), (-102.5, 40.5), (-101.5, 40.5), (-101.5, 39.5), (-102.5, 39.5)]]}, 'type': 'Feature',
                                     'id': '325', 'properties': properties})

        with fiona.open(os.path.join(output_folder, ops.prefix + '_ugid.shp')) as f:
            rows = list(f)
            fiona_meta = deepcopy(f.meta)
            fiona_crs = fiona_meta.pop('crs')
            self.assertEqual(CoordinateReferenceSystem(value=fiona_crs), WGS84())
            self.assertDictEqual(fiona_meta, {'driver': u'ESRI Shapefile', 'schema': {'geometry': 'Polygon',
                                                                                      'properties': OrderedDict(
                                                                                          [(u'UGID', 'int:10'), (
                                                                                              u'STATE_NAME',
                                                                                              'str:80')])}})
            self.assertEqual(rows, [{'geometry': {'type': 'Polygon', 'coordinates': [
                [(-101.407393, 40.001003), (-102.051535, 39.998918), (-102.047545, 40.342644), (-102.04762, 40.431077),
                 (-102.046031, 40.697319), (-102.046992, 40.74313), (-102.047739, 40.998071), (-102.621257, 41.000214),
                 (-102.652271, 40.998124), (-103.382956, 41.000316), (-103.572316, 40.999648), (-104.051705, 41.003211),
                 (-104.054012, 41.388085), (-104.0555, 41.564222), (-104.053615, 41.698218), (-104.053513, 41.999815),
                 (-104.056219, 42.614669), (-104.056199, 43.003062), (-103.501464, 42.998618), (-103.005875, 42.999354),
                 (-102.788384, 42.995303), (-102.086701, 42.989887), (-101.231737, 42.986843), (-100.198142, 42.991095),
                 (-99.53279, 42.992335), (-99.253971, 42.992389), (-98.497651, 42.991778), (-98.457444, 42.93716),
                 (-98.391204, 42.920135), (-98.310339, 42.881794), (-98.167826, 42.839571), (-98.144869, 42.835794),
                 (-98.123117, 42.820223), (-98.12182, 42.80836), (-98.03314, 42.769192), (-97.995144, 42.766812),
                 (-97.963558, 42.77369), (-97.929477, 42.792324), (-97.889941, 42.831271), (-97.888659, 42.855807),
                 (-97.818643, 42.866587), (-97.797028, 42.849597), (-97.772186, 42.846164), (-97.72525, 42.858008),
                 (-97.685752, 42.836837), (-97.63497, 42.861285), (-97.570654, 42.84799), (-97.506132, 42.860136),
                 (-97.483159, 42.857157), (-97.457263, 42.850443), (-97.389306, 42.867433), (-97.311414, 42.861771),
                 (-97.271457, 42.850014), (-97.243189, 42.851826), (-97.224443, 42.841202), (-97.211831, 42.812573),
                 (-97.161422, 42.798619), (-97.130469, 42.773923), (-97.015139, 42.759542), (-96.979593, 42.758313),
                 (-96.970003, 42.752065), (-96.977869, 42.727308), (-96.970773, 42.721147), (-96.908234, 42.731699),
                 (-96.81014, 42.704084), (-96.810437, 42.681341), (-96.799344, 42.670019), (-96.722658, 42.668592),
                 (-96.69906, 42.657715), (-96.694596, 42.641163), (-96.715273, 42.621907), (-96.714059, 42.612302),
                 (-96.636672, 42.550731), (-96.629294, 42.522693), (-96.605467, 42.507236), (-96.584753, 42.518287),
                 (-96.547215, 42.520499), (-96.494701, 42.488459), (-96.439394, 42.48924), (-96.396074, 42.467401),
                 (-96.39789, 42.441793), (-96.417628, 42.414777), (-96.411761, 42.380918), (-96.424175, 42.349279),
                 (-96.389781, 42.328789), (-96.3687, 42.298023), (-96.342881, 42.282081), (-96.332658, 42.260307),
                 (-96.337708, 42.229522), (-96.363512, 42.214042), (-96.352165, 42.168185), (-96.285123, 42.123452),
                 (-96.265483, 42.048897), (-96.238725, 42.028438), (-96.236093, 42.001258), (-96.202842, 41.996615),
                 (-96.185217, 41.980685), (-96.147328, 41.966254), (-96.14587, 41.924907), (-96.15997, 41.904151),
                 (-96.135623, 41.86262), (-96.076417, 41.791469), (-96.099321, 41.752975), (-96.099771, 41.731563),
                 (-96.085557, 41.704987), (-96.122202, 41.694913), (-96.120264, 41.684094), (-96.099306, 41.65468),
                 (-96.111307, 41.599006), (-96.080835, 41.576), (-96.091936, 41.563145), (-96.08584, 41.537522),
                 (-96.050172, 41.524335), (-96.004592, 41.536663), (-95.993965, 41.528103), (-95.996688, 41.511517),
                 (-96.013451, 41.492994), (-96.006897, 41.481954), (-95.953185, 41.472387), (-95.935065, 41.462381),
                 (-95.940056, 41.394805), (-95.942895, 41.340077), (-95.889107, 41.301389), (-95.897591, 41.286863),
                 (-95.911202, 41.308469), (-95.93023, 41.302056), (-95.910981, 41.225245), (-95.92225, 41.207854),
                 (-95.9161, 41.194063), (-95.859198, 41.180537), (-95.859801, 41.166865), (-95.876685, 41.164202),
                 (-95.858274, 41.109187), (-95.878804, 41.065871), (-95.859539, 41.035002), (-95.860897, 41.00265),
                 (-95.837603, 40.974258), (-95.836541, 40.901108), (-95.834396, 40.8703), (-95.846435, 40.848332),
                 (-95.85179, 40.7926), (-95.876616, 40.730436), (-95.767999, 40.643117), (-95.757546, 40.620904),
                 (-95.767479, 40.589048), (-95.763412, 40.549707), (-95.737036, 40.532373), (-95.692066, 40.524129),
                 (-95.687413, 40.56117), (-95.675693, 40.565835), (-95.662944, 40.558729), (-95.65806, 40.530332),
                 (-95.68497, 40.512205), (-95.695361, 40.485338), (-95.636817, 40.39639), (-95.634185, 40.3588),
                 (-95.616201, 40.346497), (-95.617933, 40.331418), (-95.645553, 40.322346), (-95.646827, 40.309109),
                 (-95.595532, 40.309776), (-95.547137, 40.266215), (-95.476822, 40.226855), (-95.466636, 40.213255),
                 (-95.460952, 40.173995), (-95.422476, 40.131743), (-95.392813, 40.115416), (-95.384542, 40.095362),
                 (-95.403784, 40.080379), (-95.413764, 40.048111), (-95.390532, 40.04375), (-95.371244, 40.028751),
                 (-95.345067, 40.024974), (-95.308697, 39.999407), (-95.329701, 39.992595), (-95.7807, 39.993489),
                 (-96.001253, 39.995159), (-96.240598, 39.994503), (-96.454038, 39.994172), (-96.80142, 39.994476),
                 (-96.908287, 39.996154), (-97.361912, 39.99738), (-97.816589, 39.999729), (-97.929588, 39.998452),
                 (-98.264165, 39.998434), (-98.504479, 39.997129), (-98.720632, 39.998461), (-99.064747, 39.998338),
                 (-99.178201, 39.999577), (-99.627859, 40.002987), (-100.18091, 40.000478), (-100.191111, 40.000585),
                 (-100.735049, 39.999172), (-100.754856, 40.000198), (-101.322148, 40.001821),
                 (-101.407393, 40.001003)]]}, 'type': 'Feature', 'id': '0',
                                     'properties': OrderedDict([(u'UGID', 1), (u'STATE_NAME', u'Nebraska')])}, {
                                        'geometry': {'type': 'Polygon', 'coordinates': [
                                            [(-95.071931, 37.001478), (-95.406622, 37.000615), (-95.526019, 37.001018),
                                             (-95.785748, 36.998114), (-95.957961, 37.000083), (-96.006049, 36.998333),
                                             (-96.519187, 37.000577), (-96.748696, 37.000166), (-97.137693, 36.999808),
                                             (-97.465405, 36.996467), (-97.80425, 36.998567), (-98.104529, 36.998671),
                                             (-98.347143, 36.999061), (-98.540219, 36.998376), (-98.999516, 36.998072),
                                             (-99.437473, 36.994558), (-99.544639, 36.995463), (-99.999261, 36.995417),
                                             (-100.088574, 36.997652), (-100.634245, 36.997832),
                                             (-100.950587, 36.996661),
                                             (-101.071604, 36.997466), (-101.553676, 36.996693),
                                             (-102.024519, 36.988875),
                                             (-102.037207, 36.988994), (-102.04201, 37.386279),
                                             (-102.044456, 37.641474),
                                             (-102.043976, 37.734398), (-102.046061, 38.253822),
                                             (-102.045549, 38.263343),
                                             (-102.047584, 38.615499), (-102.047568, 38.69255),
                                             (-102.048972, 39.037003),
                                             (-102.047874, 39.126753), (-102.048801, 39.562803),
                                             (-102.049442, 39.568693),
                                             (-102.051535, 39.998918), (-101.407393, 40.001003),
                                             (-101.322148, 40.001821),
                                             (-100.754856, 40.000198), (-100.735049, 39.999172),
                                             (-100.191111, 40.000585),
                                             (-100.18091, 40.000478), (-99.627859, 40.002987), (-99.178201, 39.999577),
                                             (-99.064747, 39.998338), (-98.720632, 39.998461), (-98.504479, 39.997129),
                                             (-98.264165, 39.998434), (-97.929588, 39.998452), (-97.816589, 39.999729),
                                             (-97.361912, 39.99738), (-96.908287, 39.996154), (-96.80142, 39.994476),
                                             (-96.454038, 39.994172), (-96.240598, 39.994503), (-96.001253, 39.995159),
                                             (-95.7807, 39.993489), (-95.329701, 39.992595), (-95.308697, 39.999407),
                                             (-95.240961, 39.942105), (-95.207597, 39.938176), (-95.193963, 39.91018),
                                             (-95.150551, 39.908054), (-95.100722, 39.869865), (-95.063246, 39.866538),
                                             (-95.033506, 39.877844), (-95.021772, 39.896978), (-94.965023, 39.900823),
                                             (-94.938243, 39.896081), (-94.936511, 39.849386), (-94.923876, 39.833131),
                                             (-94.898324, 39.828332), (-94.888505, 39.8174), (-94.899323, 39.793775),
                                             (-94.933267, 39.782773), (-94.935114, 39.775426), (-94.9218, 39.757841),
                                             (-94.877067, 39.760679), (-94.871185, 39.754118), (-94.87786, 39.739305),
                                             (-94.905678, 39.726755), (-94.930856, 39.727026), (-94.953142, 39.736501),
                                             (-94.961786, 39.732038), (-94.97857, 39.684988), (-95.028292, 39.661913),
                                             (-95.056017, 39.625689), (-95.053613, 39.586776), (-95.108988, 39.560692),
                                             (-95.102037, 39.532848), (-95.047599, 39.485328), (-95.040511, 39.46294),
                                             (-94.986204, 39.439461), (-94.958494, 39.411447), (-94.925748, 39.381266),
                                             (-94.898281, 39.38064), (-94.911343, 39.340121), (-94.907681, 39.323028),
                                             (-94.881107, 39.286046), (-94.833476, 39.261766), (-94.820819, 39.211004),
                                             (-94.790049, 39.196883), (-94.730531, 39.171256), (-94.675514, 39.174922),
                                             (-94.646407, 39.158427), (-94.612653, 39.151649), (-94.601224, 39.141227),
                                             (-94.608137, 39.112801), (-94.609281, 39.044667), (-94.612469, 38.837109),
                                             (-94.613148, 38.737222), (-94.618717, 38.471473), (-94.619053, 38.392032),
                                             (-94.61733, 38.055784), (-94.616735, 38.030387), (-94.619293, 37.679869),
                                             (-94.618996, 37.650374), (-94.618764, 37.360766), (-94.618977, 37.327732),
                                             (-94.620664, 37.060147), (-94.620379, 36.997046), (-95.032745, 37.000779),
                                             (-95.071931, 37.001478)]]}, 'type': 'Feature', 'id': '1',
                                        'properties': OrderedDict([(u'UGID', 2), (u'STATE_NAME', u'Kansas')])}])

        # # test aggregation
        ops = OcgOperations(dataset=self.get_dataset(),
                            geom='states',
                            output_format='shp',
                            aggregate=True,
                            prefix='aggregation_clip',
                            spatial_operation='clip')
        ret = ops.execute()

        with fiona.open(ret) as f:
            self.assertEqual(f.meta['schema']['properties']['GID'], 'int:10')
            rows = list(f)
        for row in rows:
            self.assertEqual(row['properties']['UGID'], row['properties']['GID'])
        self.assertEqual(set([row['properties']['GID'] for row in rows]), set([1, 2]))
        self.assertEqual(len(rows), 244)
        self.assertEqual(set(os.listdir(os.path.join(self.current_dir_output, ops.prefix))), set(
            ['aggregation_clip_ugid.shp', 'aggregation_clip.cpg', 'aggregation_clip_metadata.txt',
             'aggregation_clip_did.csv', 'aggregation_clip.log', 'aggregation_clip.dbf', 'aggregation_clip.shx',
             'aggregation_clip_ugid.prj', 'aggregation_clip_ugid.cpg', 'aggregation_clip_ugid.shx',
             'aggregation_clip.shp', 'aggregation_clip_ugid.dbf', 'aggregation_clip.prj',
             'aggregation_clip_source_metadata.txt']))

    def test_csv_conversion(self):
        ocgis.env.OVERWRITE = True
        ops = OcgOperations(dataset=self.get_dataset(), output_format='csv')
        ret = self.get_ret(ops)

        # # test with a geometry to check writing of user-geometry overview shapefile
        geom = make_poly((38, 39), (-104, -103))
        ops = OcgOperations(dataset=self.get_dataset(), output_format='csv', geom=geom)
        ret = ops.execute()

        output_dir = os.path.join(self.current_dir_output, ops.prefix)
        contents = set(os.listdir(output_dir))
        self.assertEqual(contents, set(
            ['ocgis_output_source_metadata.txt', 'ocgis_output_metadata.txt', 'ocgis_output.log',
             'ocgis_output_did.csv', 'ocgis_output.csv']))
        with open(ret, 'r') as f:
            reader = csv.DictReader(f)
            row = reader.next()
            self.assertDictEqual(row, {'LID': '1', 'UGID': '1', 'VID': '1', 'ALIAS': 'foo', 'DID': '1', 'YEAR': '2000',
                                       'VALUE': '1.0', 'MONTH': '3', 'VARIABLE': 'foo', 'GID': '6',
                                       'TIME': '2000-03-01 12:00:00', 'TID': '1', 'LEVEL': '50', 'DAY': '1'})

        did_file = os.path.join(output_dir, ops.prefix + '_did.csv')
        uri = os.path.join(self.current_dir_output, self.fn)
        with open(did_file, 'r') as f:
            reader = csv.DictReader(f)
            row = reader.next()
            self.assertDictEqual(row, {'ALIAS': 'foo', 'DID': '1', 'URI': uri, 'UNITS': 'K',
                                       'STANDARD_NAME': 'Maximum Temperature Foo', 'VARIABLE': 'foo',
                                       'LONG_NAME': 'foo_foo'})

        with open(ret, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        ops = OcgOperations(dataset=self.get_dataset(), output_format='numpy', geom=geom)
        npy = ops.execute()
        self.assertEqual(len(rows), reduce(lambda x, y: x * y, npy.gvu(1, 'foo').shape))

    def test_csv_calc_conversion(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        calc_grouping = ['month', 'year']
        ops = OcgOperations(dataset=self.get_dataset(), output_format='csv', calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()

        with open(ret, 'r') as f:
            reader = csv.DictReader(f)
            row = reader.next()
            self.assertDictEqual(row, {'LID': '1', 'UGID': '1', 'VID': '1', 'CID': '1', 'DID': '1', 'YEAR': '2000',
                                       'TIME': '2000-03-16 00:00:00', 'CALC_ALIAS': 'my_mean', 'VALUE': '1.0',
                                       'MONTH': '3', 'VARIABLE': 'foo', 'ALIAS': 'foo', 'GID': '1', 'CALC_KEY': 'mean',
                                       'TID': '1', 'LEVEL': '50', 'DAY': '16'})

    def test_csv_calc_conversion_two_calculations(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}, {'func': 'min', 'name': 'my_min'}]
        calc_grouping = ['month', 'year']
        d1 = self.get_dataset()
        d1['alias'] = 'var1'
        d2 = self.get_dataset()
        d2['alias'] = 'var2'
        ops = OcgOperations(dataset=[d1, d2], output_format='csv', calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()

        with open(ret, 'r') as f:
            with open(os.path.join(self.path_bin, 'test_csv_calc_conversion_two_calculations.csv')) as f2:
                reader = csv.DictReader(f)
                reader2 = csv.DictReader(f2)
                for row, row2 in zip(reader, reader2):
                    self.assertDictEqual(row, row2)

    def test_calc_multivariate_conversion(self):
        rd1 = self.get_dataset()
        rd1['alias'] = 'var1'
        rd2 = self.get_dataset()
        rd2['alias'] = 'var2'
        calc = [{'name': 'divide', 'func': 'divide', 'kwds': {'arr1': 'var1', 'arr2': 'var2'}}]

        for o in OutputFormat.iter_possible():
            calc_grouping = ['month']

            try:
                ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=calc_grouping, output_format=o,
                                    prefix=o + 'yay')
            except DefinitionValidationError:
                self.assertEqual(o, 'esmpy')
                continue

            ret = ops.execute()

            if o in [constants.OUTPUT_FORMAT_CSV, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]:
                with open(ret, 'r') as f:
                    reader = csv.DictReader(f)
                    row = reader.next()
                    self.assertDictEqual(row,
                                         {'LID': '1', 'UGID': '1', 'CID': '1', 'LEVEL': '50', 'DID': '', 'YEAR': '2000',
                                          'TIME': '2000-03-16 00:00:00', 'CALC_ALIAS': 'divide', 'VALUE': '1.0',
                                          'MONTH': '3', 'GID': '1', 'CALC_KEY': 'divide', 'TID': '1', 'DAY': '16'})

            if o == 'nc':
                with nc_scope(ret) as ds:
                    self.assertIn('divide', ds.variables)
                    self.assertTrue(np.all(ds.variables['divide'][:] == 1.))

            if o == 'shp':
                with fiona.open(ret) as f:
                    row = f.next()
                    self.assertIn('CID', row['properties'])

    def test_meta_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_METADATA)
        self.get_ret(ops)

    def test_csv_shapefile_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_CSV_SHAPEFILE)
        ops.execute()

        geom = make_poly((38, 39), (-104, -103))
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_CSV_SHAPEFILE, geom=geom,
                            prefix='with_ugid')
        ops.execute()

        path = os.path.join(self.current_dir_output, 'with_ugid')
        contents = os.listdir(path)
        self.assertEqual(set(contents), set(
            ['with_ugid_metadata.txt', 'with_ugid.log', 'with_ugid.csv', 'with_ugid_source_metadata.txt', 'shp',
             'with_ugid_did.csv']))

        shp_path = os.path.join(path, 'shp')
        contents = os.listdir(shp_path)
        self.assertEqual(set(contents), set(
            ['with_ugid_gid.dbf', 'with_ugid_gid.prj', 'with_ugid_ugid.shx', 'with_ugid_gid.shp', 'with_ugid_ugid.prj',
             'with_ugid_ugid.cpg', 'with_ugid_ugid.shp', 'with_ugid_gid.cpg', 'with_ugid_gid.shx',
             'with_ugid_ugid.dbf']))

        gid_path = os.path.join(shp_path, 'with_ugid_gid.shp')
        with fiona.open(gid_path) as f:
            to_test = list(f)
        self.assertEqual(to_test, [{'geometry': {'type': 'Polygon', 'coordinates': [
            [(-104.5, 37.5), (-104.5, 38.5), (-103.5, 38.5), (-103.5, 37.5), (-104.5, 37.5)]]}, 'type': 'Feature',
                                    'id': '0', 'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 6)])}, {
                                       'geometry': {'type': 'Polygon', 'coordinates': [
                                           [(-103.5, 37.5), (-103.5, 38.5), (-102.5, 38.5), (-102.5, 37.5),
                                            (-103.5, 37.5)]]}, 'type': 'Feature', 'id': '1',
                                       'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 7)])}, {
                                       'geometry': {'type': 'Polygon', 'coordinates': [
                                           [(-104.5, 38.5), (-104.5, 39.5), (-103.5, 39.5), (-103.5, 38.5),
                                            (-104.5, 38.5)]]}, 'type': 'Feature', 'id': '2',
                                       'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 10)])}, {
                                       'geometry': {'type': 'Polygon', 'coordinates': [
                                           [(-103.5, 38.5), (-103.5, 39.5), (-102.5, 39.5), (-102.5, 38.5),
                                            (-103.5, 38.5)]]}, 'type': 'Feature', 'id': '3',
                                       'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 11)])}])

        ugid_path = os.path.join(shp_path, 'with_ugid_ugid.shp')
        with fiona.open(ugid_path) as f:
            to_test = list(f)
            fiona_meta = deepcopy(f.meta)
            fiona_crs = fiona_meta.pop('crs')
            self.assertEqual(CoordinateReferenceSystem(value=fiona_crs), WGS84())
            self.assertEqual(fiona_meta, {'driver': u'ESRI Shapefile', 'schema': {'geometry': 'Polygon',
                                                                                  'properties': OrderedDict(
                                                                                      [(u'UGID', 'int:10')])}})
        self.assertEqual(to_test, [{'geometry': {'type': 'Polygon', 'coordinates': [
            [(-104.0, 38.0), (-104.0, 39.0), (-103.0, 39.0), (-103.0, 38.0), (-104.0, 38.0)]]}, 'type': 'Feature',
                                    'id': '0', 'properties': OrderedDict([(u'UGID', 1)])}])


class TestSimpleNoSpatialBounds(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcNoSpatialBounds
    fn = 'test_simple_spatial_no_bounds_01.nc'

    def test_interpolate_bounds(self):
        ret = self.get_ops(kwds={'interpolate_spatial_bounds': False}).execute()
        self.assertIsNone(ret[1]['foo'].spatial.geom.polygon)

        ret = self.get_ops(kwds={'interpolate_spatial_bounds': True}).execute()
        polygons = ret[1]['foo'].spatial.geom.polygon.value
        self.assertIsInstance(polygons[0, 0], Polygon)


class TestSimpleMask(TestSimpleBase):
    base_value = None
    nc_factory = SimpleMaskNc
    fn = 'test_simple_mask_spatial_01.nc'

    def test_spatial(self):
        self.return_shp = False
        ret = self.get_ret()
        ref = ret[1][self.var].variables[self.var].value.mask
        cmp = np.array([[True, False, False, False],
                        [False, False, False, True],
                        [False, False, False, False],
                        [True, True, False, False]])
        for tidx, lidx in itertools.product(range(0, ref.shape[0]), range(ref.shape[1])):
            self.assertTrue(np.all(cmp == ref[tidx, lidx, :]))

        # aggregation
        ret = self.get_ret(kwds={'aggregate': True})
        ref = ret[1][self.var].variables[self.var]
        self.assertAlmostEqual(ref.value.mean(), 2.58333333333, 5)
        ref = ret[1][self.var]
        self.assertEqual(ref.spatial.uid.shape, (1, 1))

    def test_empty_mask(self):
        geom = make_poly((37.762, 38.222), (-102.281, -101.754))
        with self.assertRaises(exc.MaskedDataError):
            self.get_ret(kwds={'geom': geom, 'output_format': 'shp'})
        self.get_ret(kwds={'geom': geom, 'allow_empty': True})
        with self.assertRaises(exc.MaskedDataError):
            self.get_ret(kwds={'geom': geom, 'output_format': 'numpy'})
        ret = self.get_ret(kwds={'geom': geom, 'output_format': 'numpy', 'allow_empty': True})
        self.assertTrue(ret[1]['foo'].variables['foo'].value.mask.all())


class TestSimpleMultivariate(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcMultivariate
    fn = 'test_simple_multivariate_01.nc'
    var = ['foo', 'foo2']

    def get_request_dataset(self, **kwargs):
        ds = self.get_dataset(**kwargs)
        return RequestDataset(**ds)

    def get_multiple_request_datasets(self):
        rd1 = self.get_request_dataset()
        rd1.name = 'rd1'
        rd1.alias = ['f1', 'f2']
        rd2 = self.get_request_dataset()
        rd2.name = 'rd2'
        rd2.alias = ['ff1', 'ff2']
        return [rd1, rd2]

    def run_field_tst(self, field):
        self.assertEqual([v.name for v in field.variables.itervalues()], ['foo', 'foo2'])
        self.assertEqual(field.variables.values()[0].value.mean(), 2.5)
        self.assertEqual(field.variables.values()[1].value.mean(), 5.5)
        sub = field[:, 3, 1, 1:3, 1:3]
        self.assertNumpyAll(sub.variables.values()[0].value.data.flatten(), np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertNumpyAll(sub.variables.values()[1].value.data.flatten(), np.array([1.0, 2.0, 3.0, 4.0]) + 3)

    def test_field(self):
        rd = self.get_request_dataset()
        self.assertEqual(rd.name, 'foo_foo2')
        self.run_field_tst(rd.get())

    def test_operations_convert_numpy(self):
        name = [None, 'custom_name']
        for n in name:
            n = n or '_'.join(self.var)
            rd = self.get_request_dataset()
            rd.name = n
            ops = OcgOperations(dataset=rd, output_format='numpy')
            ret = ops.execute()
            self.run_field_tst(ret[1][rd.name])

    def test_operations_convert_numpy_multiple_request_datasets(self):
        rds = self.get_multiple_request_datasets()
        ops = OcgOperations(dataset=rds, output_format='numpy')
        ret = ops.execute()
        self.assertEqual(set(ret[1].keys()), set(['rd1', 'rd2']))
        self.assertEqual(ret[1]['rd1'].variables.keys(), ['f1', 'f2'])
        self.assertEqual(ret[1]['rd2'].variables.keys(), ['ff1', 'ff2'])
        for row in ret.get_iter_melted():
            self.run_field_tst(row['field'])

    def test_operations_convert_nc_one_request_dataset(self):
        rd = self.get_request_dataset()
        ops = OcgOperations(dataset=rd, output_format='nc')
        ret = ops.execute()
        out_folder = os.path.split(ret)[0]
        csv_path = os.path.join(out_folder, 'ocgis_output_did.csv')
        with nc_scope(ret) as ds:
            self.assertEqual(ds.variables['foo'][:].mean(), 2.5)
            self.assertEqual(ds.variables['foo2'][:].mean(), 5.5)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            lines = list(reader)
        for row in lines:
            self.assertEqual(row.pop('URI'), os.path.join(self.current_dir_output, self.fn))
        actual = [
            {'ALIAS': 'foo', 'DID': '1', 'UNITS': 'K', 'STANDARD_NAME': 'Maximum Temperature Foo', 'VARIABLE': 'foo',
             'LONG_NAME': 'foo_foo'},
            {'ALIAS': 'foo2', 'DID': '1', 'UNITS': 'mm/s', 'STANDARD_NAME': 'Precipitation Foo', 'VARIABLE': 'foo2',
             'LONG_NAME': 'foo_foo_pr'}]
        for a, l in zip(actual, lines):
            self.assertDictEqual(a, l)

    def test_operations_convert_multiple_request_datasets(self):
        for o in OutputFormat.iter_possible():
            if o in [constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH, constants.OUTPUT_FORMAT_NETCDF]:
                continue
            rds = self.get_multiple_request_datasets()
            try:
                ops = OcgOperations(dataset=rds, output_format=o, prefix=o, slice=[None, [0, 2], None, None, None])
            except DefinitionValidationError:
                # only one dataset for esmpy output
                self.assertEqual(o, 'esmpy')
                continue
            ret = ops.execute()
            path_source_metadata = os.path.join(self.current_dir_output, ops.prefix,
                                                '{0}_source_metadata.txt'.format(ops.prefix))
            if o not in ['numpy', 'meta']:
                self.assertTrue(os.path.exists(ret))
                with open(path_source_metadata, 'r') as f:
                    lines = f.readlines()
                search = re.search('URI = (.*)\n', lines[1]).groups()[0]
                self.assertTrue(os.path.exists(search))

    def test_calculation_multiple_request_datasets(self):
        rds = self.get_multiple_request_datasets()
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rds, calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()
        self.assertEqual(set(ret[1].keys()), set(['rd1', 'rd2']))
        for row in ret.get_iter_melted():
            self.assertEqual(row['variable'].name, 'mean')
            self.assertEqual(row['field'].shape, (1, 2, 2, 4, 4))
            try:
                self.assertEqual(row['variable'].value.mean(), 2.5)
            except AssertionError:
                if row['variable'].parents.values()[0].name == 'foo2':
                    self.assertEqual(row['variable'].value.mean(), 5.5)
                else:
                    raise


class TestSimple360(TestSimpleBase):
    fn = 'test_simple_360_01.nc'
    nc_factory = SimpleNc360

    def test_select_geometries_wrapped(self):
        """
        Selection geometries should be wrapped appropriately when used to subset a 0-360 dataset when vector wrap is
        False.
        """

        def _get_is_wrapped_(bounds):
            return np.any(np.array(bounds) < 0)

        geom = wkt.loads(
            'POLYGON((267.404839 39.629032,268.767204 39.884946,269.775806 39.854839,269.926344 39.305376,269.753226 38.665591,269.647849 38.319355,269.128495 37.980645,268.654301 37.717204,267.954301 37.626882,267.961828 37.197849,267.269355 37.769892,266.968280 38.959140,266.968280 38.959140,266.968280 38.959140,267.404839 39.629032))')
        keywords = dict(vector_wrap=[False, True],
                        output_format=[constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_SHAPEFILE])

        for ctr, k in enumerate(itr_products_keywords(keywords, as_namedtuple=True)):
            dct = self.get_dataset()
            rd = RequestDataset(**dct)
            ops = OcgOperations(dataset=rd, vector_wrap=k.vector_wrap, output_format=k.output_format, geom=geom,
                                prefix=str(ctr))
            ret = ops.execute()
            if k.output_format == constants.OUTPUT_FORMAT_SHAPEFILE:
                path_ugid = ret.replace('.', '_ugid.')
                rows = list(ShpCabinetIterator(path=path_ugid))
                bounds = rows[0]['geom'].bounds
            else:
                bounds = ret.geoms[1].bounds
            if k.vector_wrap:
                self.assertTrue(_get_is_wrapped_(bounds))
            else:
                self.assertFalse(_get_is_wrapped_(bounds))

    def test_vector_wrap_in_operations(self):
        """Test output is appropriately wrapped."""

        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        self.assertEqual(field.spatial.wrapped_state, WrappableCoordinateReferenceSystem._flag_unwrapped)
        ops = OcgOperations(dataset=rd, vector_wrap=True)
        ret = ops.execute()
        self.assertEqual(ret[1]['foo'].spatial.wrapped_state, WrappableCoordinateReferenceSystem._flag_wrapped)

    def test_wrap(self):

        def _get_longs_(geom):
            ret = np.array([g.centroid.x for g in geom.flat])
            return (ret)

        ret = self.get_ret(kwds={'vector_wrap': False})
        longs_unwrap = _get_longs_(ret[1][self.var].spatial.abstraction_geometry.value)
        self.assertTrue(np.all(longs_unwrap > 180))

        ret = self.get_ret(kwds={'vector_wrap': True})
        longs_wrap = _get_longs_(ret[1][self.var].spatial.abstraction_geometry.value)
        self.assertTrue(np.all(np.array(longs_wrap) < 180))

        self.assertTrue(np.all(longs_unwrap - 360 == longs_wrap))

    def test_spatial_touch_only(self):
        geom = [make_poly((38.2, 39.3), (-93, -92)), make_poly((38, 39), (-93.1, -92.1))]

        for abstraction, g in itertools.product(['polygon', 'point'], geom):
            try:
                ops = self.get_ops(kwds={'geom': g, 'abstraction': abstraction})
                ret = ops.execute()
                self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), 4)
                self.get_ret(kwds={'vector_wrap': False})
                ret = self.get_ret(kwds={'geom': g, 'vector_wrap': False, 'abstraction': abstraction})
                self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), 4)
            except ExtentError:
                if abstraction == 'point':
                    pass
                else:
                    raise

    def test_spatial(self):
        geom = make_poly((38.1, 39.1), (-93.1, -92.1))

        for abstraction in ['polygon', 'point']:
            n = 1 if abstraction == 'point' else 4
            ops = self.get_ops(kwds={'geom': geom, 'abstraction': abstraction})
            ret = ops.execute()
            self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), n)
            self.get_ret(kwds={'vector_wrap': False})
            ret = self.get_ret(kwds={'geom': geom, 'vector_wrap': False, 'abstraction': abstraction})
            self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), n)


class TestSimpleProjected(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcProjection
    fn = 'test_simple_spatial_projected_01.nc'

    def test_differing_projection_no_output_crs(self):
        nc_normal = SimpleNc()
        nc_normal.write()
        uri = os.path.join(self.current_dir_output, nc_normal.filename)

        rd_projected = self.get_dataset()
        rd_projected['alias'] = 'projected'
        rd_normal = {'uri': uri, 'variable': 'foo', 'alias': 'normal'}
        dataset = [rd_projected, rd_normal]

        output_format = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_SHAPEFILE,
                         constants.OUTPUT_FORMAT_NETCDF, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]
        for o in output_format:
            try:
                OcgOperations(dataset=dataset, output_format=o)
            except DefinitionValidationError:
                if o != constants.OUTPUT_FORMAT_NUMPY:
                    pass


    def test_differing_projection_with_output_crs(self):
        nc_normal = SimpleNc()
        nc_normal.write()
        uri = os.path.join(self.current_dir_output, nc_normal.filename)

        rd_projected = self.get_dataset()
        rd_projected['alias'] = 'projected'
        rd_normal = {'uri': uri, 'variable': 'foo', 'alias': 'normal'}
        dataset = [rd_projected, rd_normal]

        output_format = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_SHAPEFILE,
                         constants.OUTPUT_FORMAT_NETCDF, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]

        for o in output_format:
            try:
                ops = OcgOperations(dataset=dataset, output_format=o, output_crs=CFWGS84(),
                                    prefix=o)
                ret = ops.execute()

                if o == constants.OUTPUT_FORMAT_NUMPY:
                    uids = []
                    for field in ret[1].itervalues():
                        uids.append(field.uid)
                        self.assertIsInstance(field.spatial.crs, CFWGS84)
                    self.assertEqual(set(uids), set([1, 2]))
                if o == constants.OUTPUT_FORMAT_SHAPEFILE:
                    with fiona.open(ret) as f:
                        self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), CFWGS84())
                        aliases = set([row['properties']['ALIAS'] for row in f])
                    self.assertEqual(set(['projected', 'normal']), aliases)
                if o == constants.OUTPUT_FORMAT_CSV_SHAPEFILE:
                    with open(ret, 'r') as f:
                        reader = csv.DictReader(f)
                        collect = {'dids': [], 'aliases': []}
                        for row in reader:
                            collect['dids'].append(int(row['DID']))
                            collect['aliases'].append(row['ALIAS'])
                        self.assertEqual(set(['projected', 'normal']), set(collect['aliases']))
                        self.assertEqual(set([1, 2]), set(collect['dids']), msg='did missing in csv file')

                    gid_shp = os.path.join(ops.dir_output, ops.prefix, 'shp', ops.prefix + '_gid.shp')
                    with fiona.open(gid_shp) as f:
                        dids = set([row['properties']['DID'] for row in f])
                        self.assertEqual(dids, set([1, 2]), msg='did missing in overview file')

            except DefinitionValidationError:
                if o == constants.OUTPUT_FORMAT_NETCDF:
                    pass
                else:
                    raise

    def test_nc_projection(self):
        dataset = self.get_dataset()
        ret = self.get_ret(kwds={'output_format': 'nc'})
        self.assertNcEqual(dataset['uri'], ret,
                           ignore_attributes={'global': ['history'], 'time_bnds': ['calendar', 'units'],
                                              'crs': ['proj4', 'units']})

    def test_nc_projection_to_shp(self):
        ret = self.get_ret(kwds={'output_format': constants.OUTPUT_FORMAT_SHAPEFILE})
        with fiona.open(ret) as f:
            self.assertEqual(f.meta['crs']['proj'], 'lcc')

    def test_with_geometry(self):
        self.get_ret(kwds={'output_format': constants.OUTPUT_FORMAT_SHAPEFILE, 'prefix': 'as_polygon'})

        features = [
            {'NAME': 'a',
             'wkt': 'POLYGON((-425985.928175 -542933.565515,-425982.789465 -542933.633257,-425982.872261 -542933.881644,-425985.837852 -542933.934332,-425985.837852 -542933.934332,-425985.928175 -542933.565515))'},
            {'NAME': 'b',
             'wkt': 'POLYGON((-425982.548605 -542936.839709,-425982.315272 -542936.854762,-425982.322799 -542936.937558,-425982.526024 -542936.937558,-425982.548605 -542936.839709))'},
        ]

        from_crs = RequestDataset(**self.get_dataset()).get().spatial.crs
        to_sr = CoordinateReferenceSystem(epsg=4326).sr
        for feature in features:
            geom = wkt.loads(feature['wkt'])
            geom = project_shapely_geometry(geom, from_crs.sr, to_sr)
            feature['wkt'] = geom.wkt

        path = os.path.join(self.current_dir_output, 'ab_{0}.shp'.format('polygon'))
        with FionaMaker(path, geometry='Polygon') as fm:
            fm.write(features)
        ocgis.env.DIR_SHPCABINET = self.current_dir_output

        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_SHAPEFILE,
                            geom='ab_polygon')
        ret = ops.execute()
        ugid_shp = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')

        with fiona.open(ugid_shp) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), from_crs)

        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_SHAPEFILE,
                            geom='ab_polygon', output_crs=CFWGS84(), prefix='xx')
        ret = ops.execute()
        ugid_shp = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')

        with fiona.open(ugid_shp) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), WGS84())
        with fiona.open(ret) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), WGS84())
