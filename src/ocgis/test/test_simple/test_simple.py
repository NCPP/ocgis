import abc
import csv
import datetime
import itertools
import os.path
from abc import abstractproperty
from collections import OrderedDict
from copy import deepcopy

import fiona
import netCDF4 as nc
import numpy as np
import six
from fiona.crs import from_string
from nose.plugins.skip import SkipTest
from shapely import wkt
from shapely.geometry.geo import mapping, shape
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

import ocgis
from ocgis import RequestDataset, vm
from ocgis import exc, env, constants
from ocgis import osr
from ocgis.base import get_variable_names
from ocgis.collection.field import Field
from ocgis.collection.spatial import SpatialCollection
from ocgis.constants import WrappedState, KeywordArgument, TagName, HeaderName
from ocgis.exc import ExtentError, DefinitionValidationError
from ocgis.ops.core import OcgOperations
from ocgis.ops.interpreter import OcgInterpreter
from ocgis.ops.parms.definition import OutputFormat
from ocgis.ops.parms.definition import SpatialOperation
from ocgis.spatial.geom_cabinet import GeomCabinetIterator
from ocgis.spatial.grid import Grid
from ocgis.test import strings
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.test.test_simple.make_test_data import SimpleNcNoLevel, SimpleNc, SimpleMaskNc, \
    SimpleNc360, SimpleNcProjection, SimpleNcNoSpatialBounds, SimpleNcMultivariate
from ocgis.util.helpers import make_poly, project_shapely_geometry
from ocgis.util.itester import itr_products_keywords
from ocgis.variable import crs
from ocgis.variable.base import Variable
from ocgis.variable.crs import CoordinateReferenceSystem, WGS84
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.temporal import TemporalVariable
from ocgis.vmachine.mpi import MPI_SIZE, MPI_RANK, MPI_COMM, get_standard_comm_state


@six.add_metaclass(abc.ABCMeta)
class TestSimpleBase(TestBase):
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
        if MPI_RANK == 0:
            self.nc_factory().write()
        MPI_COMM.Barrier()

    def get_dataset(self, time_range=None, level_range=None, time_region=None):
        uri = os.path.join(env.DIR_OUTPUT, self.fn)
        return ({'uri': uri, 'variable': self.var,
                 'time_range': time_range, 'level_range': level_range,
                 'time_region': time_region})

    def get_request_dataset(self, *args, **kwargs):
        if KeywordArgument.URI not in kwargs:
            kwargs[KeywordArgument.URI] = os.path.join(env.DIR_OUTPUT, self.fn)
        return RequestDataset(*args, **kwargs)

    def get_ops(self, kwds=None, time_range=None, level_range=None, root=0):
        if kwds is None:
            kwds = {}
        else:
            kwds = deepcopy(kwds)

        dataset = self.get_dataset(time_range, level_range)
        if KeywordArgument.OUTPUT_FORMAT not in kwds:
            kwds.update({KeywordArgument.OUTPUT_FORMAT: constants.OutputFormatName.OCGIS})
        kwds.update({KeywordArgument.DATASET: dataset})
        if KeywordArgument.DIR_OUTPUT not in kwds:
            if MPI_RANK == root:
                dir_output = self.current_dir_output
            else:
                dir_output = None
            dir_output = MPI_COMM.bcast(dir_output, root=root)
            kwds[KeywordArgument.DIR_OUTPUT] = dir_output
        ops = OcgOperations(**kwds)

        return ops

    def get_ret(self, ops=None, kwds={}, shp=False, time_range=None, level_range=None, root=0):
        """
        :param ops:
        :type ops: :class:`ocgis.driver.operations.OcgOperations`
        :param dict kwds:
        :param bool shp: If ``True``, override output format to shapefile.
        :param time_range:
        :type time_range: list[:class:`datetime.datetime`]
        :param level_range:
        :type level_range: list[int]
        """

        if ops is None:
            ops = self.get_ops(kwds, time_range=time_range, level_range=level_range, root=root)
        self.ops = ops

        ret = OcgInterpreter(ops).execute()

        if shp or self.return_shp:
            kwds2 = kwds.copy()
            kwds2.update({KeywordArgument.OUTPUT_FORMAT: constants.OutputFormatName.SHAPEFILE})
            ops2 = OcgOperations(**kwds2)
            OcgInterpreter(ops2).execute()

        return ret

    def make_shp(self):
        ops = OcgOperations(dataset=self.dataset, output_format=constants.OutputFormatName.SHAPEFILE)
        OcgInterpreter(ops).execute()


@attr('simple')
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


@attr('simple')
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

        ugeom = 'POLYGON((-104.000538 39.004301,-102.833871 39.215054,-102.833871 39.215054,-102.833871 39.215054,' \
                '-102.879032 37.882796,-104.136022 37.867742,-104.000538 39.004301))'
        ugeom = wkt.loads(ugeom)
        from_sr = osr.SpatialReference()
        from_sr.ImportFromEPSG(4326)
        to_sr = osr.SpatialReference()
        to_sr.ImportFromProj4('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 '
                              '+datum=NAD83 +units=m +no_defs')
        ugeom = project_shapely_geometry(ugeom, from_sr, to_sr)
        crs = from_string(to_sr.ExportToProj4())

        geom = [{'geom': ugeom, 'crs': crs}]
        ops = OcgOperations(dataset=rd, geom=geom)
        ret = ops.execute()

        to_test = ret.get_element()
        to_test = to_test.get_field_slice({'time': 0, 'level': 0}, strict=False)
        to_test = to_test[self.var].get_masked_value()
        actual = np.ma.array([[[[1.0, 2.0], [3.0, 4.0]]]], mask=[[[[False, False], [False, False]]]])
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

            actual = ret.get_element(variable_name='foo').shape
            self.assertEqual(actual, (61, 2, 1, 1))
            actual = ret.get_element().grid.get_point().get_value()[0, 0]
            self.assertTrue(actual.almost_equals(Point(-104.0, 39.0)))

    def test_optimizations_in_calculations(self):
        # Pass optimizations to the calculation engine using operations and ensure the output values are equivalent.
        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        tgd = field.temporal.get_grouping(['month'])
        optimizations = {'tgds': {rd.field_name: tgd}}
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            optimizations=optimizations)
        ret_with_optimizations = ops.execute()
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            optimizations=None)
        ret_without_optimizations = ops.execute()
        t1 = ret_with_optimizations.get_element(variable_name='mean')
        t2 = ret_without_optimizations.get_element(variable_name='mean')
        self.assertNumpyAll(t1.get_value(), t2.get_value())

    def test_optimizations_in_calculations_bad_calc_grouping(self):
        # Bad calculations groupings in the optimizations should be caught and raise a value error.
        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        tgd1 = field.temporal.get_grouping('all')
        tgd2 = field.temporal.get_grouping(['year', 'month'])
        tgd3 = field.temporal.get_grouping([[3]])
        for t in [tgd1, tgd2, tgd3]:
            optimizations = {'tgds': {rd.field_name: t}}
            calc = [{'func': 'mean', 'name': 'mean'}]
            calc_grouping = ['month']
            ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                                optimizations=optimizations)
            with self.assertRaises(ValueError):
                ops.execute()

    def test_optimized_bbox_subset(self):
        rd = RequestDataset(**self.get_dataset())
        geom = [-104.4, 37.6, -102.9, 39.4]

        ops = OcgOperations(dataset=rd, geom=geom, optimized_bbox_subset=True)
        ret = ops.execute()

        grid = ret.get_element().grid
        actual = grid.get_value_stacked().tolist()
        desired = [[[38.0, 38.0], [39.0, 39.0]], [[-104.0, -103.0], [-104.0, -103.0]]]
        self.assertEqual(actual, desired)
        self.assertTrue(grid._point_name not in grid.parent)
        self.assertTrue(grid._polygon_name not in grid.parent)
        self.assertIsNone(grid.get_mask())

    def test_operations_abstraction_used_for_subsetting(self):
        """Test points are configured for subsetting when changing the high-level operations abstraction."""

        ret = self.get_ret(kwds={'abstraction': 'point'})
        ref = ret.get_element()
        self.assertEqual(ref.grid_abstraction, 'point')

        actual = ref.grid.get_abstraction_geometry().get_value()[0, 0]
        self.assertIsInstance(actual, Point)

    def test_to_csv_shp_and_shape_with_point_subset(self):
        rd = RequestDataset(**self.get_dataset())
        geom = [-103.5, 38.5]
        for of in [constants.OutputFormatName.CSV_SHAPEFILE, constants.OutputFormatName.SHAPEFILE]:
            ops = ocgis.OcgOperations(dataset=rd, geom=geom, output_format=of, prefix=of, search_radius_mult=2.0)
            ops.execute()

    def test_overwrite_add_auxiliary_files(self):
        output_formats = [constants.OutputFormatName.NETCDF, constants.OutputFormatName.CSV]
        for of in output_formats:
            # If overwrite is true, we should be able to write the netCDF output multiple times.
            env.OVERWRITE = True
            kwds = {'output_format': of, 'add_auxiliary_files': False, 'prefix': of}
            self.get_ret(kwds=kwds)
            self.get_ret(kwds=kwds)
            # Switching the argument to false will result in an error when attempting to overwrite.
            env.OVERWRITE = False
            with self.assertRaises(IOError):
                self.get_ret(kwds=kwds)

    def test_units_calendar_on_time_bounds(self):
        """Test units and calendar are copied to the time bounds."""

        rd = self.get_dataset()
        ops = ocgis.OcgOperations(dataset=rd, output_format=constants.OutputFormatName.NETCDF)
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

    def test_multiple_request_datasets(self):
        aliases = ['foo1', 'foo2', 'foo3', 'foo4']
        rds = []
        for alias in aliases:
            rd = self.get_dataset()
            rd[KeywordArgument.FIELD_NAME] = alias
            rds.append(rd)

        # Check all request datasets are accounted for in the output spatial collection.
        ops = ocgis.OcgOperations(dataset=rds)
        ret = ops.execute()
        self.assertEqual(list(ret.children[None].children.keys()), aliases)

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

        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, output_format='shp', agg_selection=True,
                            snippet=True)
        ret = ops.execute()
        ugid_path = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')
        with fiona.open(ugid_path) as f:
            self.assertEqual(len(f), 1)

        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, agg_selection=True)
        ret = ops.execute()
        ret.get_element().set_abstraction_geom()
        self.assertEqual(ret.get_element().geom.shape, (4, 4))
        self.assertEqual(len(ret.children), 1)

        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, output_format='nc', prefix='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            ref = ds.variables['foo']
            self.assertFalse(ref[:].mask.all())

        ops = OcgOperations(dataset=self.get_dataset(), agg_selection=True)
        ops.execute()

    def test_point_subset(self):
        ops = self.get_ops(kwds={'geom': [-103.5, 38.5, ], 'search_radius_mult': 2.0})
        self.assertEqual(type(ops.geom[0].geom.get_value()[0]), Point)
        ret = ops.execute()
        ref = ret.get_element()
        self.assertEqual(ref.grid.shape, (4, 4))

        ops = self.get_ops(kwds={'geom': [-103, 38, ], 'search_radius_mult': 0.01})
        ret = ops.execute()
        ref = ret.get_element()
        ref.set_abstraction_geom()
        self.assertEqual(ref.grid.shape, (1, 1))
        self.assertTrue(ref.geom.get_value()[0, 0].intersects(ops.geom[0].geom.get_value()[0]))

        ops = self.get_ops(kwds={'geom': [-103, 38, ], 'abstraction': 'point', 'search_radius_mult': 0.01})
        ret = ops.execute()
        ref = ret.get_element()
        ref.set_abstraction_geom()
        self.assertEqual(ref.grid.shape, (1, 1))
        # This is a point abstraction. Polygons are not available.
        self.assertEqual(ref.geom.geom_type, 'Point')

    def test_slicing(self):
        ops = self.get_ops(kwds={'slice': [None, None, 0, [0, 2], [0, 2]]})
        ret = ops.execute()
        ref = ret.get_element(variable_name=self.var)
        self.assertTrue(np.all(ref.get_value().flatten() == 1.0))
        self.assertEqual(ref.shape, (61, 1, 2, 2))

        ops = self.get_ops(kwds={'slice': [0, None, None, [1, 3], [1, 3]]})
        ret = ops.execute()
        ref = ret.get_element(variable_name='foo').get_value()
        self.assertTrue(np.all(np.array([1., 2., 3., 4.] == ref[0, 0, :].flatten())))

        # Pass only three slices.
        with self.assertRaises(DefinitionValidationError):
            self.get_ops(kwds={'slice': [None, [1, 3], [1, 3]]})

    def test_file_only(self):
        ret = self.get_ret(
            kwds={'output_format': 'nc', 'file_only': True, 'calc': [{'func': 'mean', 'name': 'my_mean'}],
                  'calc_grouping': ['month']})

        ds = nc.Dataset(ret, 'r')
        try:
            self.assertTrue(ds.variables['my_mean'][:].mask.all())
            actual = set(ds.variables['my_mean'].ncattrs())
            self.assertEqual(actual, {'units', 'long_name', 'standard_name', 'grid_mapping'})
        finally:
            ds.close()

        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'file_only': True, 'output_format': 'shp'})

        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'file_only': True})

    def test_return_all(self):
        """Test some basic characteristics of the returned object used for simple testing."""

        ret = self.get_ret()

        # Confirm size of geometry and grid arrays.
        ref = ret.get_element(self.var)
        ref.set_abstraction_geom()
        shps = [ref.geom, ref.grid, ref.geom]
        for attr in shps:
            self.assertEqual(attr.shape, (4, 4))

        # Confirm value array.
        ref = ret.get_element(self.var)
        desired = OrderedDict(
            [('time', (61,)), ('time_bnds', (61, 2)), ('level', (2,)), ('level_bnds', (2, 2)), ('longitude', (4,)),
             ('latitude', (4,)), ('bounds_longitude', (4, 2)), ('bounds_latitude', (4, 2)), ('foo', (61, 2, 4, 4)),
             ('ocgis_polygon', (4, 4))])
        self.assertEqual(ref.shapes, desired)
        for tidx, lidx in itertools.product(list(range(0, 61)), list(range(0, 2))):
            sub = ref.get_field_slice({'time': tidx, 'level': lidx})
            idx = self.base_value == sub[self.var].get_value()
            self.assertTrue(np.all(idx))

    def test_aggregate(self):
        ret = self.get_ret(kwds={'aggregate': True})

        # Test area weighting.
        ref = ret.get_element(variable_name=self.var)
        self.assertEqual(ref.get_value().flatten()[0], np.mean(self.base_value))

        # Test geometry reduction.
        ref = ret.get_element()
        self.assertEqual(ref.geom.shape, (1,))

    def test_time_level_subset(self):
        ret = self.get_ret(time_range=[datetime.datetime(2000, 3, 1),
                                       datetime.datetime(2000, 3, 31, 23)],
                           level_range=[50, 50])
        ref = ret.get_element(self.var)
        desired = {'time_bnds': (31, 2), 'bounds_latitude': (4, 2), 'bounds_longitude': (4, 2),
                   'level': (1,), 'level_bnds': (1, 2), 'longitude': (4,), 'time': (31,), 'latitude': (4,),
                   'foo': (31, 1, 4, 4)}
        self.assertDictEqual(ref.shapes, desired)

    def test_time_level_subset_aggregate(self):
        ret = self.get_ret(kwds={'aggregate': True},
                           time_range=[datetime.datetime(2000, 3, 1), datetime.datetime(2000, 3, 31)],
                           level_range=[50, 50], )
        ref = ret.get_element(variable_name=self.var).get_masked_value()
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))
        ref = ret.get_element()
        self.assertEqual(ref.level.get_value().shape, (1,))

    def test_time_region_subset(self):
        """Test subsetting a Field object by a time region."""

        rd = ocgis.RequestDataset(uri=os.path.join(env.DIR_OUTPUT, self.fn),
                                  variable=self.var)
        ops = ocgis.OcgOperations(dataset=rd)
        ret = ops.execute()
        all = ret.get_element('foo').time.value_datetime

        def get_ref(month, year):
            rd = ocgis.RequestDataset(uri=os.path.join(env.DIR_OUTPUT, self.fn),
                                      variable=self.var,
                                      time_region={'month': month, 'year': year})
            ops = ocgis.OcgOperations(dataset=rd)
            ret = ops.execute()
            ref = ret.get_element('foo').time.value_datetime
            months = set([i.month for i in ref.flat])
            years = set([i.year for i in ref.flat])
            if month is not None:
                for m in month:
                    self.assertTrue(m in months)
            if year is not None:
                for m in year:
                    self.assertTrue(m in years)
            return ref

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
        ret = self.get_ret(kwds={'output_format': constants.OutputFormatName.OCGIS, 'geom': poly,
                                 'prefix': 'subset', 'spatial_operation': 'clip', 'aggregate': True})
        actual = ret.get_element(variable_name=self.var).get_value().mean()
        self.assertEqual(actual, 2.5)

    def test_spatial(self):
        # Intersects operation.
        geom = make_poly((37.5, 39.5), (-104.5, -102.5))
        ret = self.get_ret(kwds={'geom': geom})
        ret = ret.get_element()
        self.assertEqual(ret[self.var].shape, (61, 2, 2, 2))
        actual = ret[self.var].get_value()[0, 0, :, :].tolist()
        desired = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(actual, desired)

        # Intersection operation.
        geom = make_poly((38, 39), (-104, -103))
        ret = self.get_ret(kwds={'geom': geom, 'spatial_operation': 'clip'})
        ret = ret.get_element()
        self.assertEqual(ret[self.var].shape, (61, 2, 2, 2))
        intersection_areas = [g.area for g in ret.geom.get_value().flat]
        self.assertAlmostEqual(np.mean(intersection_areas), 0.25)

        # Intersection + Aggregation.
        geom = make_poly((38, 39), (-104, -103))
        ret = self.get_ret(kwds={'geom': geom, 'spatial_operation': 'clip', 'aggregate': True})
        ref = ret.get_element()
        self.assertEqual(ref.geom.shape, (1,))
        self.assertAlmostEqual(ref.geom.get_value()[0].area, 1.0)
        self.assertAlmostEqual(ref[self.var].get_value().mean(), 2.5)

    def test_empty_intersects(self):
        geom = make_poly((20, 25), (-90, -80))

        with self.assertRaises(exc.ExtentError):
            self.get_ret(kwds={'geom': geom})

        ret = self.get_ret(kwds={'geom': geom, 'allow_empty': True})
        # Empty fields are not added to the collection. The selection geometry should have not subsetted fields
        # associated with it in a collection.
        self.assertEqual(len(ret.children[1].children), 1)

    def test_empty_time_subset(self):
        ds = self.get_dataset(time_range=[datetime.datetime(2900, 1, 1), datetime.datetime(3100, 1, 1)])

        ops = OcgOperations(dataset=ds)
        with self.assertRaises(ExtentError):
            ops.execute()

        ops = OcgOperations(dataset=ds, allow_empty=True)
        ret = ops.execute()
        self.assertEqual(len(ret.get_element('foo').children), 0)

    def test_snippet(self):
        sc = self.get_ret(kwds={'snippet': True})
        field = sc.get_element(self.var)
        self.assertEqual(field.time.shape, (1,))
        desired = {'time_bnds': (1, 2), 'bounds_latitude': (4, 2), 'bounds_longitude': (4, 2), 'level': (1,),
                   'level_bnds': (1, 2), 'longitude': (4,), 'time': (1,), 'latitude': (4,),
                   'foo': (1, 1, 4, 4)}
        self.assertDictEqual(field.shapes, desired)
        with nc_scope(os.path.join(self.current_dir_output, self.fn)) as ds:
            to_test = ds.variables['foo'][0, 0, :, :].reshape(1, 1, 4, 4)
            self.assertNumpyAll(to_test, field['foo'].get_value())

        calc = [{'func': 'mean', 'name': 'my_mean'}]
        group = ['month', 'year']
        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'calc': calc, 'calc_grouping': group, 'snippet': True})

    def test_snippet_time_region(self):
        # Snippet is not implemented for time regions.
        with self.assertRaises(DefinitionValidationError):
            rd = self.get_dataset(time_region={'month': [1]})
            OcgOperations(dataset=rd, snippet=True)

    def test_calc(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        group = ['month', 'year']

        ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': group})
        ref = ret.get_element('foo', variable_name='my_mean')
        self.assertEqual(ref.shape, (2, 2, 4, 4))

        # Test with a spatial aggregation.
        for calc_raw in [False, True]:
            ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': group, 'aggregate': True, 'calc_raw': calc_raw})
            ref = ret.get_element('foo', variable_name='my_mean')
            self.assertEqual(ref.shape, (2, 2, 1))
            self.assertEqual(ref.get_value().flatten().mean(), 2.5)
            actual = ret.children[None].children['foo']['my_mean'].attrs
            self.assertDictEqual(actual, {'long_name': 'Mean', 'standard_name': 'mean', 'units': 'K'})

    def test_calc_multivariate(self):
        rd1 = self.get_dataset()
        rd1[KeywordArgument.RENAME_VARIABLE] = 'var1'
        rd2 = self.get_dataset()
        rd2[KeywordArgument.RENAME_VARIABLE] = 'var2'
        calc = [{'name': 'divide', 'func': 'divide', 'kwds': {'arr1': 'var1', 'arr2': 'var2'}}]

        calc_grouping = [None, ['month']]
        for cg in calc_grouping:
            ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=cg)
            ret = ops.execute()
            actual = ret.get_element(variable_name='divide').shape
            if cg is None:
                self.assertEqual(actual, (61, 2, 4, 4))
            else:
                self.assertEqual(actual, (2, 2, 4, 4))

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
        rd2[KeywordArgument.RENAME_VARIABLE] = 'foo2'
        calc = 'foo3=foo+foo2+4'
        ocgis.env.OVERWRITE = True
        for of in OutputFormat.iter_possible():
            try:
                ops = ocgis.OcgOperations(dataset=[rd, rd2], calc=calc, output_format=of,
                                          slice=[None, [0, 10], None, None, None])
            except DefinitionValidationError:
                # Only one dataset allowed for ESMPy and JSON metadata output.
                self.assertIn(of, [constants.OutputFormatName.ESMPY_GRID, constants.OutputFormatName.METADATA_JSON])
                continue
            ret = ops.execute()
            if of == constants.OutputFormatName.OCGIS:
                actual = ret.get_element(variable_name='foo3').get_value().mean()
                self.assertAlmostEqual(actual, 9.0)
            if of == 'nc':
                with nc_scope(ret) as ds:
                    self.assertEqual(ds.variables['foo3'][:].mean(), 9.0)

    def test_metadata_output(self):
        rd = RequestDataset(**self.get_dataset())
        of = constants.OutputFormatName.METADATA_OCGIS
        ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format=of)
        ret = ops.execute()
        self.assertTrue(len(ret) > 100)

    def test_nc_conversion(self):
        rd = self.get_dataset()
        ops = OcgOperations(dataset=rd, output_format='nc')
        ret = self.get_ret(ops)

        self.assertNcEqual(ret, rd['uri'], ignore_attributes={'global': ['history'],
                                                              'time_bnds': ['calendar', 'units'],
                                                              rd['variable']: ['grid_mapping'],
                                                              'time': ['axis'],
                                                              'level': ['axis'],
                                                              'latitude': ['axis', 'units', 'standard_name'],
                                                              'longitude': ['axis', 'units', 'standard_name']},
                           ignore_variables=['latitude_longitude'])

        with self.nc_scope(ret) as ds:
            expected = {'time': 'T', 'level': 'Z', 'latitude': 'Y', 'longitude': 'X'}
            for k, v in expected.items():
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

            # Time variable bounds attribute should be named to account for climatology.
            with self.assertRaises(AttributeError):
                assert ds.variables['time'].bounds

            self.assertEqual(ds.variables['time'].climatology, 'climatology_bounds')
            self.assertEqual(ds.variables['climatology_bounds'].shape, (2, 2))
        finally:
            ds.close()

    def test_nc_conversion_multiple_request_datasets(self):
        rd1 = self.get_dataset()
        rd2 = self.get_dataset()
        rd2[KeywordArgument.FIELD_NAME] = 'foo2'
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=[rd1, rd2], output_format=constants.OutputFormatName.NETCDF)

    def test_nc_conversion_level_subset(self):
        rd = self.get_dataset(level_range=[50, 50])
        ops = OcgOperations(dataset=rd, output_format='nc', prefix='one_level')
        one_level = ops.execute()

        ops = OcgOperations(dataset={'uri': one_level, 'variable': 'foo'}, output_format='nc', prefix='one_level_again')
        one_level_again = ops.execute()

        self.assertNcEqual(one_level, one_level_again, ignore_attributes={'global': ['history']})

        ds = nc.Dataset(one_level_again)
        try:
            ref = ds.variables['foo'][:]
            self.assertEqual(ref.shape[1], 1)
        finally:
            ds.close()

    def test_nc_conversion_multivariate_calculation(self):
        rd1 = self.get_dataset()
        rd2 = self.get_dataset()
        rd2[KeywordArgument.RENAME_VARIABLE] = 'foo2'
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
        ret = self.get_ret(kwds=dict(output_crs=output_crs, output_format='shp', snippet=True))
        with fiona.open(ret) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), output_crs)

    def test_geojson_projection(self):
        output_crs = CoordinateReferenceSystem(epsg=2163)
        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds=dict(output_crs=output_crs, output_format='geojson'))

    def test_empty_dataset_for_operations(self):
        with self.assertRaises(DefinitionValidationError):
            OcgOperations()

    def test_shp_conversion(self):
        ocgis.env.OVERWRITE = True
        calc = [None, [{'func': 'mean', 'name': 'my_mean'}]]
        group = ['month', 'year']

        keywords = dict(calc=calc, melted=[False, True])

        for k in self.iter_product_keywords(keywords):
            ops = OcgOperations(dataset=self.get_dataset(), output_format='shp', calc_grouping=group, calc=k.calc,
                                melted=k.melted)
            ret = self.get_ret(ops)

            if k.calc is None:
                with fiona.open(ret) as f:
                    target = f.meta['schema']['properties']
                    if k.melted:
                        schema_properties = OrderedDict(
                            [('DID', 'int:10'), ('GID', 'int:10'), ('TIME', 'str:50'), ('LB_TIME', 'str:50'),
                             ('UB_TIME', 'str:50'), ('YEAR', 'int:10'), ('MONTH', 'int:10'), ('DAY', 'int:10'),
                             ('LEVEL', 'int:10'), ('LB_LEVEL', 'int:10'), ('UB_LEVEL', 'int:10'),
                             ('VARIABLE', 'str:50'), ('VALUE', 'float:24.15')])
                    else:
                        schema_properties = OrderedDict(
                            [('DID', 'int:10'), ('GID', 'int:10'), ('TIME', 'str:50'), ('LB_TIME', 'str:50'),
                             ('UB_TIME', 'str:50'), ('YEAR', 'int:10'), ('MONTH', 'int:10'), ('DAY', 'int:10'),
                             ('LEVEL', 'int:10'), ('LB_LEVEL', 'int:10'), ('UB_LEVEL', 'int:10'),
                             ('foo', 'float:24.15')])
                    self.assertAsSetEqual(list(target.keys()), list(schema_properties.keys()))
                    fiona_meta_actual = {'crs': {'a': 6370997, 'no_defs': True, 'b': 6370997, 'proj': 'longlat'},
                                         'driver': 'ESRI Shapefile',
                                         'schema': {'geometry': 'Polygon', 'properties': schema_properties}}
                    self.assertFionaMetaEqual(f.meta, fiona_meta_actual, abs_dtype=False)
                    self.assertEqual(len(f), 1952)
            else:
                with fiona.open(ret) as f:
                    if k.melted:
                        actual = OrderedDict(
                            [('DID', 'int:10'), ('GID', 'int:10'), ('TIME', 'str:50'), ('LB_TIME', 'str:50'),
                             ('UB_TIME', 'str:50'), ('YEAR', 'int:10'), ('MONTH', 'int:10'), ('DAY', 'int:10'),
                             ('LEVEL', 'int:10'), ('LB_LEVEL', 'int:10'), ('UB_LEVEL', 'int:10'),
                             ('CALC_KEY', 'str:50'), ('SRC_VAR', 'str:50'), ('VARIABLE', 'str:50'),
                             ('VALUE', 'float:24.15')])
                    else:
                        actual = OrderedDict(
                            [('DID', 'int:10'), ('GID', 'int:10'), ('TIME', 'str:50'), ('LB_TIME', 'str:50'),
                             ('UB_TIME', 'str:50'), ('YEAR', 'int:10'), ('MONTH', 'int:10'), ('DAY', 'int:10'),
                             ('LEVEL', 'int:10'), ('LB_LEVEL', 'int:10'), ('UB_LEVEL', 'int:10'),
                             ('my_mean', 'float:24.15'), ('CALC_KEY', 'str:50'), ('SRC_VAR', 'str:50')])
                    fiona_meta_actual = {'crs': {'a': 6370997, 'no_defs': True, 'b': 6370997, 'proj': 'longlat'},
                                         'driver': 'ESRI Shapefile',
                                         'schema': {'geometry': 'Polygon', 'properties': actual}}
                    self.assertFionaMetaEqual(f.meta, fiona_meta_actual, abs_dtype=False)
                    self.assertEqual(len(f), 64)

    def test_shp_conversion_with_external_geometries(self):
        env.ENABLE_FILE_LOGGING = True

        def _make_record_(wkt_str, ugid, state_name):
            geom = wkt.loads(wkt_str)
            record = {'geometry': mapping(geom),
                      'properties': {'UGID': ugid, 'STATE_NAME': state_name}}
            return record

        nebraska = strings.S5
        kansas = strings.S6
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

        ocgis.env.DIR_GEOMCABINET = self.current_dir_output

        # Test spatial aggregation. ------------------------------------------------------------------------------------

        # OcgOperations(dataset=self.get_dataset(), output_format='shp', snippet=True, prefix='dataset').execute()
        for aggregate in [False, True]:
            ops = OcgOperations(dataset=self.get_dataset(),
                                geom='states',
                                output_format='shp',
                                aggregate=aggregate,
                                prefix='aggregation_clip_{}'.format(aggregate),
                                spatial_operation='clip',
                                melted=True)
            ret = ops.execute()

            with fiona.open(ret) as f:
                self.assertTrue(f.meta['schema']['properties']['GID'].startswith('int'))
                rows = list(f)

            if aggregate:
                desired_row_count = 244
            else:
                desired_row_count = 610
            self.assertEqual(len(rows), desired_row_count)

            if aggregate:
                for row in rows:
                    self.assertEqual(row['properties']['UGID'], row['properties']['GID'])
                self.assertEqual(set([row['properties']['GID'] for row in rows]), set([1, 2]))
            else:
                self.assertEqual(set(os.listdir(os.path.join(self.current_dir_output, ops.prefix))),
                                 {'aggregation_clip_False_ugid.shp', 'aggregation_clip_False.cpg',
                                  'aggregation_clip_False_metadata.txt',
                                  'aggregation_clip_False_did.csv', 'logs',
                                  'aggregation_clip_False.dbf',
                                  'aggregation_clip_False.shx', 'aggregation_clip_False_ugid.prj',
                                  'aggregation_clip_False_ugid.cpg',
                                  'aggregation_clip_False_ugid.shx', 'aggregation_clip_False.shp',
                                  'aggregation_clip_False_ugid.dbf',
                                  'aggregation_clip_False.prj', 'aggregation_clip_False_source_metadata.txt'})

    def test_csv_conversion(self):
        ocgis.env.OVERWRITE = True
        ocgis.env.ENABLE_FILE_LOGGING = True

        geom = make_poly((38, 39), (-104, -103))
        for melted in [True, False]:
            ops = OcgOperations(dataset=self.get_dataset(), output_format='csv', geom=geom, melted=melted, snippet=True)
            ret = ops.execute()

            output_dir = os.path.join(self.current_dir_output, ops.prefix)
            contents = set(os.listdir(output_dir))
            self.assertEqual(contents,
                             {'ocgis_output_source_metadata.txt', 'ocgis_output_metadata.txt',
                              'logs',
                              'ocgis_output_did.csv', 'ocgis_output.csv'})
            with open(ret, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                if melted:
                    actual = {'LB_LEVEL': '0', 'LEVEL': '50', 'DID': '1', 'TIME': '2000-03-01 12:00:00', 'VALUE': '1.0',
                              'MONTH': '3', 'UB_LEVEL': '100', 'LB_TIME': '2000-03-01 00:00:00',
                              'YEAR': '2000', 'VARIABLE': 'foo', 'UB_TIME': '2000-03-02 00:00:00', 'DAY': '1',
                              'UGID': '1'}
                else:
                    actual = {'UGID': '1', 'LB_LEVEL': '0', 'LEVEL': '50', 'DID': '1', 'YEAR': '2000', 'MONTH': '3',
                              'UB_LEVEL': '100', 'LB_TIME': '2000-03-01 00:00:00',
                              'TIME': '2000-03-01 12:00:00', 'foo': '1.0', 'UB_TIME': '2000-03-02 00:00:00', 'DAY': '1'}
                self.assertDictEqual(row, actual)

            did_file = os.path.join(output_dir, ops.prefix + '_did.csv')
            uri = os.path.join(self.current_dir_output, self.fn)
            with open(did_file, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                self.assertDictEqual(row, {'GROUP': '', 'LONG_NAME': 'foo_foo', 'DID': '1',
                                           'URI': uri, 'UNITS': 'K',
                                           'STANDARD_NAME': 'Maximum Temperature Foo', 'VARIABLE': 'foo'})

            with open(ret, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

    def test_csv_calc_conversion(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        calc_grouping = ['month', 'year']

        for melted in [True, False]:
            ops = OcgOperations(dataset=self.get_dataset(), output_format='csv', calc=calc, calc_grouping=calc_grouping,
                                melted=melted, prefix=str(melted))
            ret = ops.execute()

            with open(ret, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                if melted:
                    desired = {'SRC_VAR': 'foo', 'LB_LEVEL': '0', 'LEVEL': '50', 'DID': '1',
                               'TIME': '2000-03-16 00:00:00', 'VALUE': '1.0', 'MONTH': '3', 'UB_LEVEL': '100',
                               'LB_TIME': '2000-03-01 00:00:00', 'YEAR': '2000', 'VARIABLE': 'my_mean',
                               'CALC_KEY': 'mean', 'UB_TIME': '2000-04-01 00:00:00', 'DAY': '16'}
                else:
                    desired = {'SRC_VAR': 'foo', 'LB_LEVEL': '0', 'LEVEL': '50', 'DID': '1',
                               'TIME': '2000-03-16 00:00:00', 'MONTH': '3', 'my_mean': '1.0', 'UB_LEVEL': '100',
                               'LB_TIME': '2000-03-01 00:00:00', 'YEAR': '2000', 'CALC_KEY': 'mean',
                               'UB_TIME': '2000-04-01 00:00:00', 'DAY': '16'}
                self.assertDictEqual(row, desired)

    def test_csv_calc_conversion_two_calculations(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}, {'func': 'min', 'name': 'my_min'}]
        calc_grouping = ['month', 'year']
        d1 = self.get_dataset()
        d1[KeywordArgument.RENAME_VARIABLE] = 'var1'
        d2 = self.get_dataset()
        d2[KeywordArgument.RENAME_VARIABLE] = 'var2'
        ops = OcgOperations(dataset=[d1, d2], output_format='csv', calc=calc, calc_grouping=calc_grouping, melted=True)
        ret = ops.execute()

        with open(ret, 'r') as f:
            with open(os.path.join(self.path_bin, 'test_csv_calc_conversion_two_calculations.csv')) as f2:
                reader = csv.DictReader(f)
                reader2 = csv.DictReader(f2)
                for row, row2 in zip(reader, reader2):
                    self.assertDictEqual(row, row2)

    def test_calc_multivariate_conversion(self):
        rd1 = self.get_dataset()
        rd1[KeywordArgument.RENAME_VARIABLE] = 'var1'
        rd2 = self.get_dataset()
        rd2[KeywordArgument.RENAME_VARIABLE] = 'var2'
        calc = [{'name': 'divide', 'func': 'divide', 'kwds': {'arr1': 'var1', 'arr2': 'var2'}}]

        for o in OutputFormat.iter_possible():
            calc_grouping = ['month']

            try:
                ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=calc_grouping, output_format=o,
                                    prefix=o + 'yay')
            except DefinitionValidationError:
                # Only one dataset allowed for some outputs.
                self.assertIn(o, [constants.OutputFormatName.ESMPY_GRID, constants.OutputFormatName.METADATA_JSON,
                                  constants.OutputFormatName.GEOJSON])
                continue

            ret = ops.execute()

            if o in [constants.OutputFormatName.CSV, constants.OutputFormatName.CSV_SHAPEFILE]:
                with open(ret, 'r') as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    desired = {'LB_LEVEL': '0', 'LEVEL': '50', 'DID': '1', 'TIME': '2000-03-16 00:00:00', 'MONTH': '3',
                               'UB_LEVEL': '100', 'LB_TIME': '2000-03-01 00:00:00', 'YEAR': '2000',
                               'CALC_KEY': 'divide', 'UB_TIME': '2000-04-01 00:00:00', 'DAY': '16', 'divide': '1.0'}
                    if o == constants.OutputFormatName.CSV_SHAPEFILE:
                        # This will have a unique geometry identifier to link with the shapefile.
                        desired = desired.copy()
                        desired['GID'] = "1"
                    self.assertDictEqual(row, desired)

            if o == 'nc':
                with nc_scope(ret) as ds:
                    self.assertIn('divide', ds.variables)
                    self.assertTrue(np.all(ds.variables['divide'][:] == 1.))

            if o == 'shp':
                with fiona.open(ret) as f:
                    row = next(f)
                    self.assertIn('divide', row['properties'])

    def test_meta_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OutputFormatName.METADATA_OCGIS)
        self.get_ret(ops)

    def test_csv_shapefile_conversion(self):
        env.ENABLE_FILE_LOGGING = True

        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OutputFormatName.CSV_SHAPEFILE)
        ops.execute()

        geom = make_poly((38, 39), (-104, -103))
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OutputFormatName.CSV_SHAPEFILE,
                            geom=geom,
                            prefix='with_ugid')
        ops.execute()

        path = os.path.join(self.current_dir_output, 'with_ugid')
        contents = os.listdir(path)
        self.assertEqual(set(contents),
                         {'with_ugid_metadata.txt', 'logs', 'with_ugid.csv',
                          'with_ugid_source_metadata.txt',
                          'shp', 'with_ugid_did.csv'})

        shp_path = os.path.join(path, 'shp')
        contents = os.listdir(shp_path)
        self.assertEqual(set(contents),
                         {'with_ugid_gid.dbf', 'with_ugid_gid.prj', 'with_ugid_ugid.shx', 'with_ugid_gid.shp',
                          'with_ugid_ugid.prj', 'with_ugid_ugid.cpg', 'with_ugid_ugid.shp', 'with_ugid_gid.cpg',
                          'with_ugid_gid.shx', 'with_ugid_ugid.dbf'})

        gid_path = os.path.join(shp_path, 'with_ugid_gid.shp')
        with fiona.open(gid_path) as f:
            to_test = list(f)
        self.assertEqual(len(to_test), 4)
        self.assertEqual(to_test[1]['properties'], {'DID': 1, 'UGID': 1, 'GID': 2})

        ugid_path = os.path.join(shp_path, 'with_ugid_ugid.shp')
        with fiona.open(ugid_path) as f:
            to_test = list(f)
            fiona_meta = deepcopy(f.meta)
            fiona_crs = fiona_meta.pop('crs')
            a = CoordinateReferenceSystem(value=fiona_crs)
            b = env.DEFAULT_COORDSYS
            self.assertEqual(a, b)

            # Comparison of the whole schema dictionary is not possible. Some installations of Fiona produce different
            # data type lengths (i.e. int:9 v int:10)
            self.assertEqual(fiona_meta['driver'], 'ESRI Shapefile')
            fiona_meta_schema = fiona_meta['schema']
            self.assertEqual(fiona_meta_schema['geometry'], 'Polygon')
            self.assertTrue(fiona_meta_schema['properties']['UGID'].startswith('int'))

        self.assertEqual(to_test, [{'geometry': {'type': 'Polygon', 'coordinates': [
            [(-104.0, 38.0), (-104.0, 39.0), (-103.0, 39.0), (-103.0, 38.0), (-104.0, 38.0)]]}, 'type': 'Feature',
                                    'id': '0', 'properties': OrderedDict([('UGID', 1)])}])


@attr('simple')
class TestSimpleNoSpatialBounds(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcNoSpatialBounds
    fn = 'test_simple_spatial_no_bounds_01.nc'

    def test_interpolate_bounds(self):
        ret = self.get_ops(kwds={'interpolate_spatial_bounds': False}).execute()
        self.assertIsNone(ret.get_element().grid.y.bounds)

        ret = self.get_ops(kwds={'interpolate_spatial_bounds': True}).execute()
        polygons = ret.get_element().grid.get_polygon().get_value()
        self.assertIsInstance(polygons[0, 0], Polygon)


@attr('simple')
class TestSimpleMask(TestSimpleBase):
    base_value = None
    nc_factory = SimpleMaskNc
    fn = 'test_simple_mask_spatial_01.nc'

    def test_spatial(self):
        self.return_shp = False
        ret = self.get_ret()
        ref = ret.get_element(variable_name=self.var).get_mask()
        cmp = np.array([[True, False, False, False],
                        [False, False, False, True],
                        [False, False, False, False],
                        [True, True, False, False]])
        for tidx, lidx in itertools.product(list(range(0, ref.shape[0])), list(range(ref.shape[1]))):
            self.assertTrue(np.all(cmp == ref[tidx, lidx, :]))

        # Test with aggregation
        ret = self.get_ret(kwds={'aggregate': True})
        ref = ret.get_element(variable_name=self.var)
        self.assertAlmostEqual(ref.get_value().mean(), 2.58333333333, 5)


class TestSimpleMPI(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    @attr('simple', 'mpi')
    def test_basic_return(self):
        """Test returning a spatial collection with no operations."""

        rd = RequestDataset(**self.get_dataset())
        desired_field_target = rd.get()

        for _ in range(3):
            ret = self.get_ret()

            self.assertIsInstance(ret, SpatialCollection)

            # If in parallel, test one of the spatial dimensions is distributed.
            field = ret.get_element()

            for dv in field.data_variables:
                self.assertFalse(dv.has_allocated_value)

            if field.is_empty:
                self.assertGreater(MPI_RANK, 1)
                self.assertTrue(desired_field_target.is_empty)
            else:
                self.assertFalse(desired_field_target.is_empty)
                x_size, y_size = field.x.shape[0], field.y.shape[0]
                if MPI_SIZE > 1:
                    self.assertNotEqual(x_size, y_size)
                else:
                    self.assertEqual(x_size, y_size)

    @attr('simple', 'mpi')
    def test_calculation(self):
        """Test a calculation in parallel."""

        rd = RequestDataset(**self.get_dataset())
        ops = OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}], calc_grouping=['day'])
        ret = ops.execute()
        ret = ret.get_element()
        days = [d.day for d in ret.time.value_datetime]

        if not ret.is_empty:
            self.assertEqual(len(days), 31)
            self.assertAsSetEqual(days, list(range(1, 32)))
        else:
            self.assertGreater(MPI_RANK, 1)

    @attr('simple', 'mpi')
    def test_dist(self):
        """Test parallel distribution metadata."""

        if MPI_SIZE != 3:
            raise SkipTest('MPI_SIZE != 3')

        dataset = self.get_dataset()
        rd = RequestDataset(**dataset)

        # Test at least one of the dimensions is distributed.
        distributed_dimension_name = 'lon'
        for rank in range(MPI_SIZE):
            dim = rd.driver.dist.get_dimension(distributed_dimension_name, rank=rank)
            if rank == 2:
                self.assertTrue(dim.is_empty)
            else:
                self.assertFalse(dim.is_empty)
            self.assertTrue(dim.dist)

    @attr('mpi', 'simple')
    def test_snippet(self):
        """Test a snippet in parallel."""

        keywords = {KeywordArgument.OUTPUT_FORMAT: [
            constants.OutputFormatName.OCGIS,
            constants.OutputFormatName.NETCDF,
            constants.OutputFormatName.SHAPEFILE,
            constants.OutputFormatName.CSV,
            constants.OutputFormatName.CSV_SHAPEFILE
        ]}

        for ctr, k in enumerate(self.iter_product_keywords(keywords, as_namedtuple=False)):
            # if ctr != 1: continue
            # barrier_print(ctr, k)

            output_format = k[KeywordArgument.OUTPUT_FORMAT]
            ret = self.get_ret(kwds={KeywordArgument.SNIPPET: True,
                                     KeywordArgument.OUTPUT_FORMAT: output_format,
                                     KeywordArgument.PREFIX: output_format})

            # barrier_print(ret)

            if output_format == constants.OutputFormatName.OCGIS:
                # Only the first rank will be non-empty.
                if MPI_RANK in [0, 1]:
                    desired = False
                else:
                    desired = True

                actual = ret.get_element().is_empty
                self.assertEqual(actual, desired)
            else:
                if MPI_RANK == 0:
                    self.assertTrue(os.path.exists(ret))

    @attr('mpi', 'simple')
    def test_spatial_subset(self):
        comm, rank, size = get_standard_comm_state()

        with vm.scoped('field.write', [0]):
            if not vm.is_null:
                path = self.get_temporary_file_path('global.nc')

                x = Variable(name='lon', dimensions='lon', value=np.linspace(0.5, 359.5, 360))
                x.set_extrapolated_bounds('lon_bounds', 'bounds')

                y = Variable(name='lat', dimensions='lat', value=np.linspace(-89.5, 89.5, 180))
                y.set_extrapolated_bounds('lat_bounds', 'bounds')

                dtime = Dimension(name='time')
                time = TemporalVariable(name='time', dimensions=dtime, value=np.arange(31))

                extra = Variable(name='extra', dimensions='extra', value=[1.0, 2.0, 3.0])

                untouched = Variable(name='untouched', dimensions='untouched', value=[7, 8])

                field = Field(grid=Grid(x, y), time=time, variables=[extra, untouched])

                data_dimensions = ['lat', 'time', 'extra', 'lon']
                data = Variable(name='data', dimensions=data_dimensions, parent=field, dtype=np.float32)
                data.get_value()[:] = 1.0
                field.append_to_tags(TagName.DATA_VARIABLES, data)

                field.write(path)
            else:
                path = None

        path = MPI_COMM.bcast(path)

        # barrier_print(path)

        geom = [-30.25, -40.9, -20.3, 15.0]

        keywords = {KeywordArgument.OUTPUT_FORMAT: [
            constants.OutputFormatName.OCGIS,
            constants.OutputFormatName.NETCDF,
            constants.OutputFormatName.SHAPEFILE,
            constants.OutputFormatName.CSV,
            constants.OutputFormatName.CSV_SHAPEFILE
        ]}

        # ocgis.env.VERBOSE = True
        # ocgis.env.DEBUG = True
        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            # if ctr != 4: continue
            # barrier_print(ctr, k)
            output_format = getattr(k, KeywordArgument.OUTPUT_FORMAT)

            if output_format in [constants.OutputFormatName.OCGIS, constants.OutputFormatName.NETCDF]:
                snippet = False
            else:
                snippet = True

            rd = RequestDataset(path)
            ops = OcgOperations(dataset=rd, geom=geom, prefix=output_format, output_format=output_format,
                                snippet=snippet)

            dir_outputs = comm.gather(ops.dir_output)
            dir_outputs = comm.bcast(dir_outputs)
            self.assertEqual(len(set(dir_outputs)), 1)

            ret = ops.execute()

            if output_format == constants.OutputFormatName.OCGIS:
                actual_field = ret.get_element()
                desired_extent = (329.0, -41.0, 340.0, 15.0)
                with vm.scoped_by_emptyable('global_extent', actual_field):
                    if not vm.is_null:
                        actual_extent = actual_field.grid.extent_global
                        self.assertEqual(actual_extent, desired_extent)

                data_in = actual_field.data_variables[0]
                self.assertFalse(data_in.has_allocated_value)

                if data_in.is_empty:
                    self.assertIsNone(data_in.get_value())
                else:
                    self.assertEqual(data_in.get_value().mean(), 1.0)


@attr('simple')
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
        rd1._field_name = 'rd1'
        rd1._rename_variable = ['f1', 'f2']
        rd2 = self.get_request_dataset()
        rd2._field_name = 'rd2'
        rd2._rename_variable = ['ff1', 'ff2']
        return [rd1, rd2]

    def run_field_tst(self, field):
        self.assertEqual(field.data_variables[0].get_value().mean(), 2.5)
        self.assertEqual(field.data_variables[1].get_value().mean(), 5.5)
        sub = field.get_field_slice({'time': 3, 'level': 1, 'y': slice(1, 3), 'x': slice(1, 3)}, strict=False)
        self.assertNumpyAll(sub.data_variables[0].get_value().flatten(), np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertNumpyAll(sub.data_variables[1].get_value().flatten(), np.array([1.0, 2.0, 3.0, 4.0]) + 3)

    def test_field(self):
        rd = self.get_request_dataset()
        actual = rd.get()
        self.assertEqual(get_variable_names(actual.data_variables), ('foo', 'foo2'))
        self.run_field_tst(actual)

    def test_operations_convert_numpy(self):
        name = [None, 'custom_name']
        for n in name:
            if n is None:
                n = n or '_'.join(self.var)
            rd = self.get_request_dataset()
            rd._field_name = n
            ops = OcgOperations(dataset=rd, output_format=constants.OutputFormatName.OCGIS)
            ret = ops.execute()
            actual = ret.get_element()
            self.assertEqual(actual.name, n)
            self.assertEqual(get_variable_names(actual.data_variables), ('foo', 'foo2'))
            self.run_field_tst(actual)

    def test_operations_convert_numpy_multiple_request_datasets(self):
        rds = self.get_multiple_request_datasets()
        ops = OcgOperations(dataset=rds, output_format=constants.OutputFormatName.OCGIS)
        ret = ops.execute()

        actual = set(ret.children[None].children.keys())
        self.assertEqual(actual, set(['rd1', 'rd2']))

        actual = get_variable_names(ret.get_element(field_name='rd1').data_variables)
        self.assertEqual(actual, ('f1', 'f2'))

        actual = get_variable_names(ret.get_element(field_name='rd2').data_variables)
        self.assertEqual(actual, ('ff1', 'ff2'))

        for field in ret.iter_fields():
            self.run_field_tst(field)

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
            {'DID': '1', 'UNITS': 'K', 'STANDARD_NAME': 'Maximum Temperature Foo', 'VARIABLE': 'foo',
             'LONG_NAME': 'foo_foo', 'GROUP': ''},
            {'DID': '1', 'UNITS': 'mm/s', 'STANDARD_NAME': 'Precipitation Foo', 'VARIABLE': 'foo2',
             'LONG_NAME': 'foo_foo_pr', 'GROUP': ''}]
        for a, l in zip(actual, lines):
            self.assertDictEqual(a, l)

    def test_operations_convert_multiple_request_datasets(self):
        for o in OutputFormat.iter_possible():
            if o in [constants.OutputFormatName.NETCDF]:
                continue
            rds = self.get_multiple_request_datasets()
            try:
                ops = OcgOperations(dataset=rds, output_format=o, prefix=o, slice=[None, [0, 2], None, None, None],
                                    melted=True)
            except DefinitionValidationError:
                # Only one dataset allowed for these outputs.
                self.assertIn(o, [constants.OutputFormatName.ESMPY_GRID, constants.OutputFormatName.METADATA_JSON,
                                  constants.OutputFormatName.GEOJSON])
                continue
            ret = ops.execute()
            path_source_metadata = os.path.join(self.current_dir_output, ops.prefix,
                                                '{0}_source_metadata.txt'.format(ops.prefix))
            if o not in [constants.OutputFormatName.OCGIS, constants.OutputFormatName.METADATA_OCGIS]:
                self.assertTrue(os.path.exists(ret))
                with open(path_source_metadata, 'r') as f:
                    lines = f.readlines()
                self.assertTrue(len(lines) > 25)

    def test_calculation_multiple_request_datasets(self):
        rds = self.get_multiple_request_datasets()
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rds, calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()
        self.assertEqual(set(ret.children[None].children.keys()), set(['rd1', 'rd2']))
        for row in ret.iter_melted(tag=TagName.DATA_VARIABLES):
            self.assertTrue(row['variable'].name.startswith('mean'))
            self.assertEqual(row['variable'].shape, (2, 2, 4, 4))
            if row['variable'].name in ['mean_f1', 'mean_ff1']:
                self.assertEqual(row['variable'].get_value().mean(), 2.5)
            elif row['variable'].name in ['mean_f2', 'mean_ff2']:
                self.assertEqual(row['variable'].get_value().mean(), 5.5)
            else:
                self.fail()


@attr('simple')
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
            'POLYGON((267.404839 39.629032,268.767204 39.884946,269.775806 39.854839,269.926344 39.305376,269.753226 '
            '38.665591,269.647849 38.319355,269.128495 37.980645,268.654301 37.717204,267.954301 37.626882,267.961828 '
            '37.197849,267.269355 37.769892,266.968280 38.959140,266.968280 38.959140,266.968280 38.959140,267.404839 '
            '39.629032))')
        keywords = dict(vector_wrap=[False, True],
                        output_format=[constants.OutputFormatName.OCGIS, constants.OutputFormatName.SHAPEFILE])

        for ctr, k in enumerate(itr_products_keywords(keywords, as_namedtuple=True)):
            if ctr != 3: continue
            dct = self.get_dataset()
            rd = RequestDataset(**dct)

            ops = OcgOperations(dataset=rd, vector_wrap=k.vector_wrap, output_format=k.output_format, geom=geom,
                                prefix=str(ctr))
            ret = ops.execute()
            if k.output_format == constants.OutputFormatName.SHAPEFILE:
                path_ugid = ret.replace('.', '_ugid.')
                rows = list(GeomCabinetIterator(path=path_ugid))
                bounds = rows[0]['geom'].bounds
            else:
                bounds = ret.geoms[1].bounds
            if k.vector_wrap:
                self.assertTrue(_get_is_wrapped_(bounds))
            else:
                self.assertFalse(_get_is_wrapped_(bounds))

    def test_vector_wrap_in_operations(self):
        """Test vector wrapping logic through operations."""

        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        self.assertEqual(field.wrapped_state, WrappedState.UNWRAPPED)

        # Data should maintain its original wrapped state if the output format is not a vector format.
        ops = OcgOperations(dataset=rd, vector_wrap=True)
        ret = ops.execute()
        self.assertEqual(ret.get_element().wrapped_state, WrappedState.UNWRAPPED)

        # If this is a vector output format, the data should be wrapped.
        for vector_wrap in [True, False]:
            ops = OcgOperations(dataset=rd, vector_wrap=vector_wrap, output_format=constants.OutputFormatName.GEOJSON,
                                snippet=True, prefix=str(vector_wrap))
            ret = ops.execute()
            with fiona.open(ret, driver='GeoJSON') as source:
                for record in source:
                    geom = shape(record['geometry'])
                    arr = np.array(geom.exterior)[:, 0]
                    if vector_wrap:
                        self.assertTrue(np.all(arr < 0))
                    else:
                        self.assertTrue(np.all(arr > 0))

    def test_wrap(self):

        def _get_lons_(geom):
            ret = np.array([g.centroid.x for g in geom.flat])
            return ret

        ret = self.get_ret(kwds={'vector_wrap': False})
        actual = ret.get_element().grid.get_abstraction_geometry().get_value()
        lons_unwrap = _get_lons_(actual)
        self.assertTrue(np.all(lons_unwrap > 180))

        ret = self.get_ret(kwds={'spatial_wrapping': 'wrap'})
        actual = ret.get_element().grid.get_abstraction_geometry().get_value()
        lons_wrap = _get_lons_(actual)
        self.assertTrue(np.all(np.array(lons_wrap) < 180))

        self.assertTrue(np.all(lons_unwrap - 360 == lons_wrap))

    def test_spatial_touch_only(self):
        """
        Test grid spatial abstractions. Point grid abstractions should return different subsetted geometries than
        polygon grid abstractions. Point grid abstractions should also return 'touched' geometries.
        """

        geom = [make_poly((38.2, 39.3), (-93, -92)), make_poly((38, 39), (-93.1, -92.1))]
        names = ['g1', 'g2']
        geom = [GeometryVariable(value=geom[idx], name=names[idx], dimensions='g', ugid=idx, crs=env.DEFAULT_COORDSYS)
                for idx in range(len(names))]

        def _get_centroids_(coll):
            field = coll.get_element()
            geom_arr = field.grid.get_abstraction_geometry().get_value().flatten()
            return [np.array(g.centroid).tolist() for g in geom_arr]

        for ctr, (abstraction, g) in enumerate(itertools.product(['polygon', 'point'], geom)):
            ops = self.get_ops(kwds={'geom': g, 'abstraction': abstraction})
            ret = ops.execute()
            actual = _get_centroids_(ret)
            if abstraction == 'polygon':
                desired = [[268.0, 38.0], [267.0, 38.0], [268.0, 39.0], [267.0, 39.0]]
            else:
                if g.name == 'g1':
                    desired = [[268.0, 39.0], [267.0, 39.0]]
                else:
                    desired = [[267.0, 38.0], [267.0, 39.0]]

            self.assertEqual(actual, desired)
            self.get_ret(kwds={'vector_wrap': False})
            ret2 = self.get_ret(kwds={'geom': g, 'vector_wrap': False, 'abstraction': abstraction})
            actual2 = _get_centroids_(ret2)
            self.assertEqual(actual2, desired)


@attr('simple')
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
        rd_normal = {'uri': uri, 'variable': 'foo'}
        dataset = [rd_projected, rd_normal]

        output_format = [constants.OutputFormatName.OCGIS, constants.OutputFormatName.SHAPEFILE,
                         constants.OutputFormatName.NETCDF, constants.OutputFormatName.CSV_SHAPEFILE]
        for o in output_format:
            try:
                OcgOperations(dataset=dataset, output_format=o)
            except DefinitionValidationError:
                if o != constants.OutputFormatName.OCGIS:
                    pass

    def test_differing_projection_with_output_crs(self):
        nc_normal = SimpleNc()
        nc_normal.write()
        uri = os.path.join(self.current_dir_output, nc_normal.filename)

        rd_projected = self.get_dataset()
        rd_projected[KeywordArgument.RENAME_VARIABLE] = 'projected'
        rd_normal = {'uri': uri, 'variable': 'foo', KeywordArgument.RENAME_VARIABLE: 'normal'}
        dataset = [rd_projected, rd_normal]

        output_format = [constants.OutputFormatName.OCGIS, constants.OutputFormatName.SHAPEFILE,
                         constants.OutputFormatName.NETCDF, constants.OutputFormatName.CSV_SHAPEFILE]

        for o in output_format:
            try:
                ops = OcgOperations(dataset=dataset, output_format=o, output_crs=WGS84(), prefix=o, melted=True,
                                    snippet=True)
                self.assertTrue(ops.melted)
                ret = ops.execute()

                if o == constants.OutputFormatName.OCGIS:
                    uids = []
                    for field in ret.iter_fields():
                        uids.append(field.uid)
                        self.assertIsInstance(field.crs, WGS84)
                    self.assertEqual(set(uids), {1, 2})
                if o == constants.OutputFormatName.SHAPEFILE:
                    with fiona.open(ret) as f:
                        self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), WGS84())
                        aliases = set([row['properties'][HeaderName.VARIABLE] for row in f])
                    self.assertEqual({'projected', 'normal'}, aliases)
                if o == constants.OutputFormatName.CSV_SHAPEFILE:
                    with open(ret, 'r') as f:
                        reader = csv.DictReader(f)
                        collect = {'dids': [], 'variables': []}
                        for row in reader:
                            collect['dids'].append(int(row['DID']))
                            collect['variables'].append(row['VARIABLE'])
                        self.assertEqual({'projected', 'normal'}, set(collect['variables']))
                        self.assertEqual({1, 2}, set(collect['dids']), msg='did missing in csv file')

                    gid_shp = os.path.join(ops.dir_output, ops.prefix, 'shp', ops.prefix + '_gid.shp')
                    with fiona.open(gid_shp) as f:
                        dids = set([row['properties']['DID'] for row in f])
                        self.assertEqual(dids, {1, 2}, msg='did missing in overview file')

            except DefinitionValidationError:
                if o == constants.OutputFormatName.NETCDF:
                    pass
                else:
                    raise

    def test_nc_projection(self):
        dataset = self.get_dataset()
        ret = self.get_ret(kwds={'output_format': 'nc'})
        self.assertNcEqual(dataset['uri'], ret,
                           ignore_attributes={'global': ['history'], 'time_bnds': ['calendar', 'units'],
                                              'crs': ['units']})

    def test_nc_projection_to_shp(self):
        ret = self.get_ret(kwds={'output_format': constants.OutputFormatName.SHAPEFILE, KeywordArgument.SNIPPET: True})
        with fiona.open(ret) as f:
            self.assertEqual(f.meta['crs']['proj'], 'lcc')

    def test_with_geometry(self):
        features = [
            {'NAME': 'a',
             'wkt': 'POLYGON((-425985.928175 -542933.565515,-425982.789465 -542933.633257,-425982.872261 -542933.881644,-425985.837852 -542933.934332,-425985.837852 -542933.934332,-425985.928175 -542933.565515))'},
            {'NAME': 'b',
             'wkt': 'POLYGON((-425982.548605 -542936.839709,-425982.315272 -542936.854762,-425982.322799 -542936.937558,-425982.526024 -542936.937558,-425982.548605 -542936.839709))'},
        ]

        from_crs = RequestDataset(**self.get_dataset()).get().crs
        to_sr = CoordinateReferenceSystem(epsg=4326).sr
        for feature in features:
            geom = wkt.loads(feature['wkt'])
            geom = project_shapely_geometry(geom, from_crs.sr, to_sr)
            feature['wkt'] = geom.wkt

        path = os.path.join(self.current_dir_output, 'ab_{0}.shp'.format('polygon'))

        geoms = [wkt.loads(f['wkt']) for f in features]
        names = [f['NAME'] for f in features]
        g = GeometryVariable(name='geom', value=geoms, crs=WGS84(), dimensions='geom')
        names = Variable(name='NAME', value=names, dimensions='geom')
        field = Field(is_data=names, geom=g, crs=WGS84())
        field.write(path, driver='vector')

        ocgis.env.DIR_GEOMCABINET = self.current_dir_output

        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OutputFormatName.SHAPEFILE,
                            geom='ab_polygon')
        ret = ops.execute()
        ugid_shp = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')

        with fiona.open(ugid_shp) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), from_crs)

        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OutputFormatName.SHAPEFILE,
                            geom='ab_polygon', output_crs=WGS84(), prefix='xx')
        ret = ops.execute()
        ugid_shp = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')

        with fiona.open(ugid_shp) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), WGS84())
        with fiona.open(ret) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), WGS84())
