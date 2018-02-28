import abc
import datetime
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy, copy
from pprint import pprint

import fiona
import netCDF4 as nc
import numpy as np
import six
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseMultipartGeometry

from ocgis import RequestDataset
from ocgis import SourcedVariable
from ocgis import Variable, Dimension
from ocgis import env
from ocgis.collection.field import Field
from ocgis.constants import GridAbstraction, KeywordArgument
from ocgis.spatial.geom_cabinet import GeomCabinet
from ocgis.spatial.grid import Grid, get_geometry_variable
from ocgis.util.helpers import get_iter, pprint_dict, get_bounds_from_1d, get_date_list, create_exact_field_value
from ocgis.util.itester import itr_products_keywords
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.temporal import TemporalVariable
from ocgis.vmachine.mpi import get_standard_comm_state, OcgDist, MPI_RANK, variable_scatter, variable_collection_scatter

"""
Definitions for various "attrs":
 * slow: long-running tests that are typically ran before a release
 * remote: tests relying on remote datasets that are typically run before a release
 * esmf: tests requiring ESMF
 * simple: simple tests designed to test core functionality requiring no datasets
 * optional: tests that use optional dependencies
 * data: test requires a data file
 * icclim: test requires ICCLIM
 * benchmark: test used for benchmarking/performance
 * cli: test related to the command line interface. requires click as a dependency.

nosetests -vs --with-id -a '!slow,!remote' ocgis
"""


class ToTest(Exception):
    """
    Useful when wanting to flag things as not tested.
    """


@six.add_metaclass(abc.ABCMeta)
class TestBase(unittest.TestCase):
    """
    All tests should inherit from this. It allows test data to be written to a temporary folder and removed easily.
    Also simplifies access to static files.
    """
    # add MPI barrier during test tear down
    add_barrier = True
    # set to false to not resent the environment before each test
    reset_env = True
    # set to false to not create and destroy a temporary directory before each test
    create_dir = True
    # Set to False to keep the test directory following test execution.
    remove_dir = True
    # set to false to not shutdown logging
    shutdown_logging = True
    # prefix for the temporary test directories
    _prefix_path_test = 'ocgis_test_'

    def __init__(self, *args, **kwargs):
        self.test_data = self.get_tst_data()
        self.current_dir_output = None
        self.ToTest = ToTest
        super(TestBase, self).__init__(*args, **kwargs)

    @property
    def path_bin(self):
        """Path to binary test file directory."""

        base_dir = os.path.realpath(os.path.split(__file__)[0])
        ret = os.path.join(base_dir, 'bin')
        return ret

    def assertAsSetEqual(self, sequence1, sequence2, msg=None):
        self.assertSetEqual(set(sequence1), set(sequence2), msg=msg)

    def assertDescriptivesAlmostEqual(self, desired_descriptives, actual_arr):
        actual_descriptives = self.create_array_descriptive_statistics(actual_arr)
        for k, v in list(desired_descriptives.items()):
            if k == 'shape':
                self.assertEqual(v, actual_descriptives['shape'])
            else:
                self.assertTrue(np.isclose(v, actual_descriptives[k]))

    def assertDictEqual(self, d1, d2, msg=None):
        """
        Asserts two dictionaries are equal. If they are not, identify the first key/value which are not equal.

        :param dict d1: A dictionary to test.
        :param dict d2: A dictionary to test.
        :param str msg: A message to attach to an assertion error.
        :raises: AssertionError
        """

        try:
            unittest.TestCase.assertDictEqual(self, d1, d2, msg=msg)
        except AssertionError:
            for k, v in d1.items():
                try:
                    msg = 'Issue with key "{0}". Values are {1}.'.format(k, (v, d2[k]))
                except KeyError:
                    msg = 'The key "{0}" was not found in the second dictionary.'.format(k)
                    raise AssertionError(msg)
                self.assertEqual(v, d2[k], msg=msg)
            self.assertEqual(set(d1.keys()), set(d2.keys()))

    def assertFionaMetaEqual(self, meta, actual, abs_dtype=True):
        self.assertEqual(CoordinateReferenceSystem(value=meta['crs']),
                         CoordinateReferenceSystem(value=actual['crs']))
        self.assertEqual(meta['driver'], actual['driver'])

        schema_meta = meta['schema']
        schema_actual = actual['schema']
        self.assertEqual(schema_meta['geometry'], schema_actual['geometry'])

        properties_meta = schema_meta['properties']
        properties_actual = schema_actual['properties']

        for km in properties_meta.keys():
            if abs_dtype:
                self.assertEqual(properties_meta[km], properties_actual[km])
            else:
                property_meta = properties_meta[km]
                property_actual = properties_actual[km]
                dtype_meta = property_meta.split(':')[0]
                dtype_actual = property_actual.split(':')[0]
                self.assertEqual(dtype_meta, dtype_actual)

    def assertIsInstances(self, actual, desired):
        self.assertTrue(isinstance, desired)

    def assertNumpyAll(self, arr1, arr2, check_fill_value=True, check_arr_dtype=True, check_arr_type=True,
                       rtol=None):
        """
        Asserts arrays are equal according to the test criteria.

        :param arr1: An array to compare.
        :type arr1: :class:`numpy.ndarray`
        :param arr2: An array to compare.
        :type arr2: :class:`numpy.ndarray`
        :param bool check_fill_value: If ``True``, check that the data type for masked array fill values are equal.
        :param bool check_arr_dtype: If ``True``, check the data types of the arrays are equal.
        :param bool check_arr_type: If ``True``, check the types of the incoming arrays.

        >>> type(arr1) == type(arr2)

        :param places: If this is a float value, use a "close" data comparison as opposed to exact comparison. The
         value is the test tolerance. See http://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html.
        :type places: float

        >>> places = 1e-4

        :raises: AssertionError
        """

        if check_arr_type:
            self.assertEqual(type(arr1), type(arr2))
        self.assertEqual(arr1.shape, arr2.shape)
        if check_arr_dtype:
            self.assertEqual(arr1.dtype, arr2.dtype)
        if isinstance(arr1, np.ma.MaskedArray) or isinstance(arr2, np.ma.MaskedArray):
            if check_fill_value:
                data_to_check = (arr1.data, arr2.data)
            else:
                data_to_check = (arr1, arr2)
            if check_arr_type:
                self.assertTrue(np.all(arr1.mask == arr2.mask))
            if check_fill_value:
                self.assertEqual(arr1.fill_value, arr2.fill_value)
        else:
            data_to_check = (arr1, arr2)

        # Check the data values.
        if rtol is None:
            to_assert = np.all(data_to_check[0] == data_to_check[1])
        else:
            to_assert = np.allclose(data_to_check[0], data_to_check[1], rtol=rtol)
        self.assertTrue(to_assert)

    def assertNumpyMayShareMemory(self, *args, **kwargs):
        self.assertTrue(np.may_share_memory(*args, **kwargs))

    def assertNcEqual(self, uri_src, uri_dest, check_types=True, close=False, metadata_only=False,
                      ignore_attributes=None, ignore_variables=None, check_fill_value=True):
        """
        Assert two netCDF files are equal according to the test criteria.

        :param str uri_src: A URI to a source file.
        :param str uri_dest: A URI to a destination file.
        :param bool check_types: If ``True``, check data types of variable arrays.
        :param bool close: If ``False``, use exact value comparisons without a tolerance.
        :param bool metadata_only: If ``False``, check array values associated with variables. If ``True``, only check
         metadata values and not value arrays.
        :param dict ignore_attributes: Select which attributes to ignore when testing. Keys are associated with variable
         names. The exception is for dataset-level attributes which are selected with the key `'global'`.

        >>> ignore_attributes = {'global': ['history']}

        :param list ignore_variables: A list of variable names to ignore.
        """

        ignore_variables = ignore_variables or []

        src = nc.Dataset(uri_src)
        dest = nc.Dataset(uri_dest)

        ignore_attributes = ignore_attributes or {}

        try:
            self.assertEqual(src.data_model, dest.data_model)

            for dimname, dim in src.dimensions.items():
                self.assertEqual(len(dim), len(dest.dimensions[dimname]))
            self.assertEqual(set(src.dimensions.keys()), set(dest.dimensions.keys()))

            for varname, var in src.variables.items():
                if varname in ignore_variables:
                    continue

                dvar = dest.variables[varname]

                var_value = var[:]
                dvar_value = dvar[:]

                try:
                    if not metadata_only:
                        if var_value.dtype == object:
                            for idx in range(var_value.shape[0]):
                                if close:
                                    self.assertNumpyAllClose(var_value[idx], dvar_value[idx])
                                else:
                                    self.assertNumpyAll(var_value[idx], dvar_value[idx], check_arr_dtype=check_types)
                        else:
                            if close:
                                self.assertNumpyAllClose(var_value, dvar_value)
                            else:
                                self.assertNumpyAll(var_value, dvar_value, check_arr_dtype=check_types,
                                                    check_fill_value=check_fill_value, check_arr_type=check_types)
                except (AssertionError, AttributeError):
                    # Some zero-length netCDF variables should not be tested for value equality. Values are meaningless
                    # and only the attributes should be tested for equality.
                    if len(dvar.dimensions) == 0:
                        self.assertEqual(len(var.dimensions), 0)
                    else:
                        raise

                if check_types:
                    self.assertEqual(var_value.dtype, dvar_value.dtype)

                # check values of attributes on all variables
                for k, v in var.__dict__.items():
                    try:
                        to_test_attr = getattr(dvar, k)
                    except AttributeError:
                        # if the variable and attribute are flagged to ignore, continue to the next attribute
                        if dvar._name in ignore_attributes:
                            if k in ignore_attributes[dvar._name]:
                                continue

                        # notify if an attribute is missing
                        msg = 'The attribute "{0}" is not found on the variable "{1}" for URI "{2}".' \
                            .format(k, dvar._name, uri_dest)
                        raise AttributeError(msg)
                    try:
                        self.assertNumpyAll(v, to_test_attr)
                    except AttributeError:
                        self.assertEqual(v, to_test_attr)

                # check values of attributes on all variables
                for k, v in dvar.__dict__.items():
                    try:
                        to_test_attr = getattr(var, k)
                    except AttributeError:
                        # if the variable and attribute are flagged to ignore, continue to the next attribute
                        if var._name in ignore_attributes:
                            if k in ignore_attributes[var._name]:
                                continue

                        # notify if an attribute is missing
                        msg = 'The attribute "{0}" is not found on the variable "{1}" for URI "{2}".' \
                            .format(k, var._name, uri_src)
                        raise AttributeError(msg)
                    try:
                        self.assertNumpyAll(v, to_test_attr)
                    except AttributeError:
                        self.assertEqual(v, to_test_attr)

                self.assertEqual(var.dimensions, dvar.dimensions)

            sets = [set(xx.variables.keys()) for xx in [src, dest]]
            for ignore_variable, s in itertools.product(ignore_variables, sets):
                try:
                    s.remove(ignore_variable)
                except KeyError:
                    # likely missing in one or the other
                    continue
            self.assertEqual(*sets)

            if 'global' not in ignore_attributes:
                self.assertDictEqual(src.__dict__, dest.__dict__)
            else:
                for k, v in src.__dict__.items():
                    if k not in ignore_attributes['global']:
                        to_test = dest.__dict__[k]
                        try:
                            self.assertNumpyAll(v, to_test)
                        except AttributeError:
                            self.assertEqual(v, to_test)
        finally:
            src.close()
            dest.close()

    def assertNumpyAllClose(self, arr1, arr2):
        """
        Asserts arrays are close according to the test criteria.

        :param arr1: An array to compare.
        :type arr1: :class:`numpy.ndarray`
        :param arr2: An array to compare.
        :type arr2: :class:`numpy.ndarray`
        :raises: AssertionError
        """

        arr1 = np.array(arr1)
        arr2 = np.array(arr2)

        self.assertEqual(type(arr1), type(arr2))
        self.assertEqual(arr1.shape, arr2.shape)
        if isinstance(arr1, np.ma.MaskedArray) or isinstance(arr2, np.ma.MaskedArray):
            self.assertTrue(np.allclose(arr1.data, arr2.data))
            self.assertTrue(np.all(arr1.mask == arr2.mask))
            self.assertEqual(arr1.fill_value, arr2.fill_value)
        else:
            self.assertTrue(np.allclose(arr1, arr2))

    def assertNumpyNotAll(self, arr1, arr2):
        """
        Asserts arrays are not equal according to the test criteria.

        :param arr1: An array to compare.
        :type arr1: :class:`numpy.ndarray`
        :param arr2: An array to compare.
        :type arr2: :class:`numpy.ndarray`
        :raises: AssertionError
        """

        try:
            self.assertNumpyAll(arr1, arr2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Arrays are equivalent.')

    def assertNumpyNotAllClose(self, arr1, arr2):
        """
        Asserts arrays are not close according to the test criteria.

        :param arr1: An array to compare.
        :type arr1: :class:`numpy.ndarray`
        :param arr2: An array to compare.
        :type arr2: :class:`numpy.ndarray`
        :raises: AssertionError
        """

        try:
            self.assertNumpyAllClose(arr1, arr2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Arrays are equivalent within precision.')

    def assertPolygonSimilar(self, lhs, rhs, places=6, check_type=True):
        """
        Assert two polygons are similar using a variety of methods. The two polygons could still differ. This avoids
        checking exact coordinates and ordering.

        :param lhs: The actual polygon.
        :type lhs: :class:`~shapely.geometry.polygon.Polygon`
        :param lhs: The desired polygon.
        :type lhs: :class:`~shapely.geometry.polygon.Polygon`
        :param int places: Numpy of decimal places to use for the fuzzy comparison.
        :param bool check_type: If ``True``, compare the types of the inputs.
        """

        self.assertAlmostEqual(lhs.area, rhs.area, places=places)
        self.assertAlmostEqual(lhs.intersection(rhs).area, lhs.area, places=places)
        if check_type:
            self.assertEqual(type(lhs), type(rhs))
        if check_type and (isinstance(lhs, BaseMultipartGeometry) or isinstance(rhs, BaseMultipartGeometry)):
            self.assertEqual(len(lhs), len(rhs))
            for l, r in zip(lhs, rhs):
                self.assertAlmostEqual(l.area, r.area, places=places)

    def assertWarns(self, warning, meth):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            meth()
            self.assertTrue(any(item.category == warning for item in warning_list))

    def assertWeightFilesEquivalent(self, global_weights_filename, merged_weights_filename):
        """Assert weight files are equivalent."""

        nwf = RequestDataset(merged_weights_filename).get()
        gwf = RequestDataset(global_weights_filename).get()
        nwf_row = nwf['row'].get_value()
        gwf_row = gwf['row'].get_value()
        self.assertAsSetEqual(nwf_row, gwf_row)
        nwf_col = nwf['col'].get_value()
        gwf_col = gwf['col'].get_value()
        self.assertAsSetEqual(nwf_col, gwf_col)
        nwf_S = nwf['S'].get_value()
        gwf_S = gwf['S'].get_value()
        self.assertEqual(nwf_S.sum(), gwf_S.sum())
        unique_src = np.unique(nwf_row)
        diffs = []
        for us in unique_src.flat:
            nwf_S_idx = np.where(nwf_row == us)[0]
            nwf_col_sub = nwf_col[nwf_S_idx]
            nwf_S_sub = nwf_S[nwf_S_idx].sum()

            gwf_S_idx = np.where(gwf_row == us)[0]
            gwf_col_sub = gwf_col[gwf_S_idx]
            gwf_S_sub = gwf_S[gwf_S_idx].sum()

            self.assertAsSetEqual(nwf_col_sub, gwf_col_sub)

            diffs.append(nwf_S_sub - gwf_S_sub)
        diffs = np.abs(diffs)
        self.assertLess(diffs.max(), 1e-14)

    @staticmethod
    def barrier_print(*args, **kwargs):
        from ocgis.vmachine.mpi import barrier_print
        barrier_print(*args, **kwargs)

    @staticmethod
    def create_array_descriptive_statistics(arr):
        ret = {'mean': arr.mean(),
               'min': arr.min(),
               'max': arr.max(),
               'std': np.std(arr),
               'shape': arr.shape}
        if arr.ndim == 1:
            arr = np.diagflat(arr)
        ret['trace'] = np.trace(arr)
        return ret

    @staticmethod
    def get_exact_field_value(longitude, latitude):
        return create_exact_field_value(longitude, latitude)

    def get_field(self, nlevel=None, nrlz=None, crs=None, ntime=2, with_bounds=False, variable_name='foo', nrow=None,
                  ncol=None):
        """
        :param int nlevel: The number of level elements.
        :param int nrlz: The number of realization elements.
        :param crs: The coordinate system for the field.
        :type crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :param ntime: The number of time elements.
        :type ntime: int
        :param with_bounds: If ``True``, extrapolate bounds on spatial dimensions.
        :type with_bounds: bool
        :param str variable_name: The field's data variable name.
        :returns: A small field object for testing.
        :rtype: `~ocgis.Field`
        """

        np.random.seed(1)
        if nrow is None:
            nrow = 2
            row_value = [4., 5.]
        else:
            row_value = range(4, 4 + nrow)
        if ncol is None:
            ncol = 2
            col_value = [40., 50.]
        else:
            col_value = range(4, 4 + ncol)
            col_value = [c * 10 for c in col_value]
        row = Variable(value=row_value, name='row', dimensions='row', dtype=float)
        col = Variable(value=col_value, name='col', dimensions='col', dtype=float)

        if with_bounds:
            row.set_extrapolated_bounds('row_bounds', 'bounds')
            col.set_extrapolated_bounds('col_bounds', 'bounds')

        grid = Grid(col, row)

        variable_dimensions = []
        variable_shape = []

        if nrlz != 0:
            if nrlz is None:
                nrlz = 1
                realization = None
            else:
                realization = Variable(value=list(range(1, nrlz + 1)), name='realization', dimensions='realization')
            variable_shape.append(nrlz)
            variable_dimensions.append('realization')
        else:
            realization = None

        if ntime == 2:
            value_temporal = [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 2, 1)]
        else:
            value_temporal = []
            start = datetime.datetime(2000, 1, 1)
            delta = datetime.timedelta(days=1)
            ctr = 0
            while ctr < ntime:
                value_temporal.append(start)
                start += delta
                ctr += 1
        tdim = Dimension('time')
        temporal = TemporalVariable(value=value_temporal, dimensions=tdim, name='time')
        variable_shape.append(temporal.shape[0])
        variable_dimensions.append('time')

        if nlevel != 0:
            if nlevel is None:
                nlevel = 1
                level = None
            else:
                level = Variable(value=list(range(1, nlevel + 1)), name='level', dimensions='level')
            variable_shape.append(nlevel)
            variable_dimensions.append('level')
        else:
            level = None

        variable_shape += [nrow, ncol]
        variable_dimensions += ['row', 'col']

        variable = Variable(name=variable_name, value=np.random.rand(*variable_shape), dimensions=variable_dimensions)
        field = Field(grid=grid, time=temporal, is_data=variable, level=level, realization=realization, crs=crs)

        return field

    def get_netcdf_path_no_dimensioned_variables(self):
        """
        :returns: A path to a small netCDF file containing no dimensioned variables.
        :rtype: str
        """

        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('dim')
            var = ds.createVariable('foovar', int, dimensions=('dim',))
            var.a_name = 'a name'
        return path

    def get_netcdf_path_no_row_column(self):
        """
        Create a NetCDF with no row and column dimensions.

        :returns: Path to the created NetCDF in the current test directory.
        :rtype: str
        """

        field = self.get_field()
        field.spatial.grid.row.set_extrapolated_bounds()
        field.spatial.grid.col.set_extrapolated_bounds()
        field.spatial.grid.value
        field.spatial.grid.corners
        self.assertIsNotNone(field.spatial.grid.corners)
        field.spatial.grid.row = field.spatial.grid.col = None
        self.assertIsNone(field.spatial.grid.row)
        self.assertIsNone(field.spatial.grid.col)
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write_netcdf(ds)
        return path

    def get_shapefile_path_with_no_ugid(self):
        """
        :returns: A path to a shapefile without the attribute "UGID". There are 11 records in the shapefile.
        :rtype: str
        """

        path = GeomCabinet().get_shp_path('state_boundaries')
        new = self.get_temporary_file_path('state_boundaries_without_ugid.shp')
        with fiona.open(path) as source:
            meta = source.meta.copy()
            meta['schema']['properties'].pop('UGID')
            with fiona.open(new, mode='w', **meta) as sink:
                for ctr, record in enumerate(source):
                    record['properties'].pop('UGID')
                    record['properties']['ID'] += 5.0
                    sink.write(record)
                    if ctr == 10:
                        break
        return new

    def get_temporary_file_path(self, name):
        """
        :param str name: The name to append to the current temporary output directory.
        :returns: Temporary path in the current output directory.
        :rtype: str
        """

        return os.path.join(self.current_dir_output, name)

    def get_temporary_output_directory(self):
        """
        :returns: A path to a temporary directory with an appropriate prefix.
        :rtype: str
        """

        ret = tempfile.mkdtemp(prefix=self._prefix_path_test)
        return ret

    def get_time_series(self, start, end):
        """
        :param start: The start date.
        :type start: :class:`datetime.datetime`
        :param end: The end date.
        :type end: :class:`datetime.datetime`
        :returns: A list of dates separated by a day.
        :rtype: list of :class:`datetime.datetime`
        """

        delta = datetime.timedelta(days=1)
        ret = []
        while start <= end:
            ret.append(start)
            start += delta
        return ret

    @staticmethod
    def get_tst_data():
        """
        :returns: A dictionary-like object with special access methods for test files.
        :rtype: :class:`ocgis.test.base.TestData`
        """

        test_data = TestData()

        test_data.update(['nc', 'CMIP3'], 'Tavg', 'Extraction_Tavg.nc', key='cmip3_extraction')
        test_data.update(['nc', 'CanCM4'], 'rhs', 'rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                         key='cancm4_rhs')
        test_data.update(['nc', 'CanCM4'], 'rhsmax', 'rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                         key='cancm4_rhsmax')
        test_data.update(['nc', 'CanCM4'], 'tas', 'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                         key='cancm4_tas')
        test_data.update(['nc', 'CanCM4'], 'tasmax', 'tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                         key='cancm4_tasmax_2001')
        test_data.update(['nc', 'CanCM4'], 'tasmax', 'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                         key='cancm4_tasmax_2011')
        test_data.update(['nc', 'CanCM4'], 'tasmin', 'tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                         key='cancm4_tasmin_2001')
        test_data.update(['nc', 'daymet'], 'tmax', 'tmax.nc', key='daymet_tmax')
        test_data.update(['nc', 'maurer', '2010'], 'pr',
                         ['nldas_met_update.obs.daily.pr.1990.nc', 'nldas_met_update.obs.daily.pr.1991.nc'],
                         key='maurer_2010_pr')
        test_data.update(['nc', 'maurer', '2010'], 'tas',
                         ['nldas_met_update.obs.daily.tas.1990.nc', 'nldas_met_update.obs.daily.tas.1991.nc'],
                         key='maurer_2010_tas')
        test_data.update(['nc', 'maurer', '2010'], 'tasmax',
                         ['nldas_met_update.obs.daily.tasmax.1990.nc', 'nldas_met_update.obs.daily.tasmax.1991.nc'],
                         key='maurer_2010_tasmax')
        test_data.update(['nc', 'maurer', '2010'], 'tasmin',
                         ['nldas_met_update.obs.daily.tasmin.1990.nc', 'nldas_met_update.obs.daily.tasmin.1991.nc'],
                         key='maurer_2010_tasmin')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'tasmax', 'Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                         key='maurer_2010_concatenated_tasmax')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'tasmin', 'Maurer02new_OBS_tasmin_daily.1971-2000.nc',
                         key='maurer_2010_concatenated_tasmin')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'tas', 'Maurer02new_OBS_tas_daily.1971-2000.nc',
                         key='maurer_2010_concatenated_tas')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'pr', 'Maurer02new_OBS_pr_daily.1971-2000.nc',
                         key='maurer_2010_concatenated_pr')
        test_data.update(['nc', 'maurer', 'bcca'], 'tasmax', 'gridded_obs.tasmax.OBS_125deg.daily.1991.nc',
                         key='maurer_bcca_1991')
        test_data.update(['nc', 'maurer', 'bccr'], 'Prcp', 'bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc',
                         key='maurer_bccr_1950')
        test_data.update(['nc', 'misc', 'month_in_time_units'], 'clt', 'clt.nc', key='clt_month_units')
        test_data.update(['nc', 'misc', 'rotated_pole'], 'pr',
                         'pr_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_mon_198101-199012.nc',
                         key='rotated_pole_cnrm_cerfacs')
        test_data.update(['nc', 'misc', 'rotated_pole'], 'tas',
                         'tas_EUR-44_CCCma-CanESM2_rcp85_r1i1p1_SMHI-RCA4_v1_sem_209012-210011.nc',
                         key='rotated_pole_cccma')
        test_data.update(['nc', 'misc', 'rotated_pole'], 'tas',
                         'tas_EUR-44_ICHEC-EC-EARTH_historical_r12i1p1_SMHI-RCA4_v1_day_19710101-19751231.nc',
                         key='rotated_pole_ichec')
        test_data.update(['nc', 'misc', 'subset_test'], 'Prcp', 'sresa2.ncar_pcm1.3.monthly.Prcp.RAW.1950-2099.nc',
                         key='subset_test_Prcp')
        test_data.update(['nc', 'misc', 'subset_test'], 'Tavg', 'Tavg_bccr_bcm2_0.1.sresa2.nc', key='subset_test_Tavg')
        test_data.update(['nc', 'misc', 'subset_test'], 'Tavg', 'sresa2.bccr_bcm2_0.1.monthly.Tavg.RAW.1950-2099.nc',
                         key='subset_test_Tavg_sresa2')
        test_data.update(['nc', 'misc', 'subset_test'], 'slp', 'slp.1955.nc', key='subset_test_slp')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_CRCM_ccsm_1981010103.nc', key='narccap_crcm')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_CRCM_ccsm_1981010103.nc', key='narccap_polar_stereographic')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_HRM3_gfdl_1981010103.nc', key='narccap_hrm3')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_RCM3_gfdl_1981010103.nc', key='narccap_rcm3')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_WRFG_ccsm_1986010103.nc', key='narccap_lambert_conformal')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_WRFG_ccsm_1986010103.nc', key='narccap_wrfg')
        test_data.update(['nc', 'narccap'], 'pr', ['pr_WRFG_ncep_1981010103.nc', 'pr_WRFG_ncep_1986010103.nc'],
                         key='narccap_pr_wrfg_ncep')
        test_data.update(['nc', 'narccap'], 'tas', 'tas_HRM3_gfdl_1981010103.nc', key='narccap_rotated_pole')
        test_data.update(['nc', 'narccap'], 'tas', 'tas_RCM3_gfdl_1981010103.nc', key='narccap_tas_rcm3_gfdl')
        test_data.update(['nc', 'QED-2013'], 'climatology_TNn_monthly_max', 'climatology_TNn_monthly_max.nc',
                         key='qed_2013_TNn_monthly_max')
        test_data.update(['nc', 'QED-2013'], 'climatology_TNn_annual_min', 'climatology_TNn_annual_min.nc',
                         key='qed_2013_TNn_annual_min')
        test_data.update(['nc', 'QED-2013'], 'climatology_TasMin_seasonal_max_of_seasonal_means',
                         'climatology_TasMin_seasonal_max_of_seasonal_means.nc',
                         key='qed_2013_TasMin_seasonal_max_of_seasonal_means')
        test_data.update(['nc', 'QED-2013'], 'climatology_Tas_annual_max_of_annual_means',
                         'climatology_Tas_annual_max_of_annual_means.nc',
                         key='qed_2013_climatology_Tas_annual_max_of_annual_means')
        test_data.update(['nc', 'QED-2013', 'multifile'], 'txxmmedm', 'maurer02v2_median_txxmmedm_january_1971-2000.nc',
                         key='qed_2013_maurer02v2_median_txxmmedm_january_1971-2000')
        test_data.update(['nc', 'QED-2013', 'multifile'], 'txxmmedm',
                         'maurer02v2_median_txxmmedm_february_1971-2000.nc',
                         key='qed_2013_maurer02v2_median_txxmmedm_february_1971-2000')
        test_data.update(['nc', 'QED-2013', 'multifile'], 'txxmmedm', 'maurer02v2_median_txxmmedm_march_1971-2000.nc',
                         key='qed_2013_maurer02v2_median_txxmmedm_march_1971-2000')
        test_data.update(['nc', 'snippets'], 'dtr', 'snippet_Maurer02new_OBS_dtr_daily.1971-2000.nc',
                         key='snippet_maurer_dtr')
        test_data.update(['nc', 'snippets'], 'bias', 'seasonalbias.nc', key='snippet_seasonalbias')

        # test_data.update(['shp', 'state_boundaries'], None, 'state_boundaries.shp', key='state_boundaries')

        return test_data

    def iter_product_keywords(self, keywords, as_namedtuple=True):
        return itr_products_keywords(keywords, as_namedtuple=as_namedtuple)

    def nautilus(self, path):
        if not os.path.isdir(path):
            path = os.path.split(path)[0]
        subprocess.call(['nautilus', path])

    def ncdump(self, path, header_only=True, variable=None):
        cmd = ['ncdump']
        if variable is not None:
            cmd.append('-v')
            cmd.append(variable)
        if header_only and variable is None:
            cmd.append('-h')
        cmd.append(path)
        subprocess.check_call(cmd)

    def inspect(self, **kwargs):
        from ocgis import RequestDataset
        rd = RequestDataset(**kwargs)
        rd.inspect()

    def nc_scope(self, *args, **kwargs):
        return nc_scope(*args, **kwargs)

    @property
    def path_state_boundaries(self):
        path_shp = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        return path_shp

    def pprint(self, *args, **kwargs):
        print('')
        pprint(*args, **kwargs)

    def pprint_dict(self, target):
        pprint_dict(target)

    def print_scope(self):
        return print_scope()

    def rank_print(self, *args, **kwargs):
        from ocgis.vmachine.mpi import rank_print
        rank_print(*args, **kwargs)

    def setUp(self):
        from ocgis import vm
        vm.__init__()

        comm, rank, size = get_standard_comm_state()

        self.current_dir_output = None
        if self.reset_env:
            env.reset()
        if self.create_dir:
            if rank == 0:
                temporary_output_directory = self.get_temporary_output_directory()
            else:
                temporary_output_directory = None
            temporary_output_directory = comm.bcast(temporary_output_directory)

            self.current_dir_output = temporary_output_directory
            env.DIR_OUTPUT = self.current_dir_output

        if self.add_barrier:
            comm.Barrier()

    def shortDescription(self):
        """
        Overloaded method so ``nose`` will not print the docstring associated with a test.
        """

        return None

    def tearDown(self):
        comm, rank, size = get_standard_comm_state()

        if self.add_barrier:
            comm.Barrier()

        try:
            if self.create_dir and self.remove_dir:
                if rank == 0:
                    shutil.rmtree(self.current_dir_output)
        finally:
            if self.reset_env:
                env.reset()
            if self.shutdown_logging:
                ocgis_lh.shutdown()

        from ocgis import vm
        vm.finalize()

        if self.add_barrier:
            comm.Barrier()


class NeedsTestData(Exception):
    pass


class TestData(OrderedDict):
    @property
    def size(self):
        """
        :returns: Size of test data in bytes.
        :rtype: int
        """

        total = 0
        for key in list(self.keys()):
            path = self.get_uri(key)
            # path is returned as a sequence...sometimes
            for element in get_iter(path):
                total += os.path.getsize(element)
        return total

    def copy_file(self, key, dest):
        """
        Copy a single files with unique test key identifier ``key`` to the full path ``dest``.

        :param str key: The unique identifier key to a test dataset.
        :param str dest: The full path for the test files to be copied to.
        """

        src = self.get_uri(key)
        dest = os.path.join(dest, self[key]['filename'])
        shutil.copy2(src, dest)
        return dest

    def get_rd(self, key, kwds=None):
        """
        :param str key: The unique identifier to the test dataset.
        :param dict kwds: Any keyword arguments to pass to :class:`ocgis.RequestDataset`
        :returns: A request dataset object to use for testing!
        :rtype: :class:`ocgis.RequestDataset`
        """
        ref = self[key]
        if kwds is None:
            kwds = {}
        kwds.update({'uri': self.get_uri(key), 'variable': ref['variable']})
        rd = RequestDataset(**kwds)
        return rd

    def get_relative_dir(self, key):
        """
        :returns: The relative directory with no starting slash.
        :rtype: str
        """

        value = self[key]
        path = os.path.join(*value['collection'])
        return path

    def get_uri(self, key):
        """
        :param str key: The unique identifier to the test dataset.
        :returns: A sequence of URIs for the test dataset selected by key.
        :rtype: list[str,] or str
        :raises: OSError, ValueError
        """
        ref = self[key]
        coll = deepcopy(ref['collection'])
        if env.DIR_TEST_DATA is None:
            raise ValueError('The TestDataset object requires env.DIR_TEST_DATA have a path value.')
        coll.insert(0, env.DIR_TEST_DATA)

        # determine if the filename is a string or a sequence of paths
        filename = ref['filename']
        if isinstance(filename, six.string_types):
            coll.append(filename)
            uri = os.path.join(*coll)
        else:
            uri = []
            for part in filename:
                copy_coll = copy(coll)
                copy_coll.append(part)
                uri.append(os.path.join(*copy_coll))

        # ensure the uris exist, if not, we may need to download
        if isinstance(uri, six.string_types):
            assert os.path.exists(uri)
        else:
            for element in uri:
                assert os.path.exists(element)

        return uri

    def update(self, collection, variable, filename, key=None):
        """
        Update storage with a new test dataset.

        :param sequence collection: A sequence of strings that when appended to the base directory will yield the full
         path to the directory containing the test dataset.

        >>> collection = ['climate_data']
        >>> collection = ['cmip', 'test_data']

        :param str variable: The variable name to extract from the dataset.
        :param str filename: The filename of the dataset.

        >>> filename = 'test_data.nc'

        :param str key: If provided, use for the unique key identifier. Otherwise, ``filename`` is used.
        """

        OrderedDict.update(self, {key or filename: {'collection': collection,
                                                    'filename': filename, 'variable': variable}})


def attr(*args, **kwargs):
    """
    Decorator that adds attributes to classes or functions for use with the Attribute (-a) plugin.

    http://nose.readthedocs.io/en/latest/plugins/attrib.html
    """

    def wrap_ob(ob):
        for name in args:
            setattr(ob, name, True)
        for name, value in kwargs.items():
            setattr(ob, name, value)
        return ob

    return wrap_ob


@contextmanager
def nc_scope(path, mode='r', format=None):
    """
    Provide a transactional scope around a :class:`netCDF4.Dataset` object.

    >>> with nc_scope('/my/file.nc') as ds:
    >>>     print ds.variables

    :param str path: The full path to the netCDF dataset.
    :param str mode: The file mode to use when opening the dataset.
    :param str format: The NetCDF format.
    :returns: An open dataset object that will be closed after leaving the ``with`` statement.
    :rtype: :class:`netCDF4.Dataset`
    """

    kwds = {'mode': mode}
    if format is not None:
        kwds['format'] = format

    ds = nc.Dataset(path, **kwds)
    try:
        yield ds
    finally:
        ds.close()


@contextmanager
def print_scope():
    class MyPrinter(object):
        def __init__(self):
            self.storage = []

        def write(self, msg):
            self.storage.append(msg)

    prev_stdout = sys.stdout
    try:
        myprinter = MyPrinter()
        sys.stdout = myprinter
        yield myprinter
    finally:
        sys.stdout = prev_stdout


@six.add_metaclass(abc.ABCMeta)
class AbstractTestInterface(TestBase):
    def assertGeometriesAlmostEquals(self, a, b):

        def _almost_equals_(a, b):
            return a.almost_equals(b)

        vfunc = np.vectorize(_almost_equals_, otypes=[bool])
        to_test = vfunc(a.value, b.value)
        self.assertTrue(to_test.all())
        self.assertNumpyAll(a.get_mask(), b.get_mask())

    def get_boundedvariable(self, mask=None):
        value = np.array([4, 5, 6], dtype=float)
        if mask is not None:
            value = np.ma.array(value, mask=mask)
        value_bounds = get_bounds_from_1d(value)
        bounds = Variable('x_bounds', value=value_bounds, dimensions=['x', 'bounds'])
        var = Variable('x', value=value, bounds=bounds, dimensions=['x'])
        return var

    def get_gridxy(self, with_2d_variables=False, crs=None, with_xy_bounds=False, with_value_mask=False,
                   with_parent=False, abstraction=GridAbstraction.AUTO):

        dest_mpi = OcgDist()
        dest_mpi.create_dimension('xdim', 3)
        dest_mpi.create_dimension('ydim', 4, dist=True)
        dest_mpi.create_dimension('time', 10)
        dest_mpi.update_dimension_bounds()

        kwds = {'crs': crs}

        if MPI_RANK == 0:
            x = [101, 102, 103]
            y = [40, 41, 42, 43]

            x_dim = Dimension('xdim', size=len(x))
            y_dim = Dimension('ydim', size=len(y))

            if with_2d_variables:
                x_value, y_value = np.meshgrid(x, y)
                x_dims = (y_dim, x_dim)
                y_dims = x_dims
            else:
                x_value, y_value = x, y
                x_dims = (x_dim,)
                y_dims = (y_dim,)

            if with_value_mask:
                x_value = np.ma.array(x_value, mask=[False, True, False])
                y_value = np.ma.array(y_value, mask=[True, False, True, False])

            vx = Variable('x', value=x_value, dtype=float, dimensions=x_dims)
            vy = Variable('y', value=y_value, dtype=float, dimensions=y_dims)
            if with_xy_bounds:
                vx.set_extrapolated_bounds('xbounds', 'bounds')
                vy.set_extrapolated_bounds('ybounds', 'bounds')
                dest_mpi.add_dimension(vx.bounds.dimensions[-1])

            if with_parent:
                np.random.seed(1)
                tas = np.random.rand(10, 3, 4)
                tas = Variable(name='tas', value=tas, dimensions=['time', 'xdim', 'ydim'])

                rhs = np.random.rand(4, 3, 10) * 100
                rhs = Variable(name='rhs', value=rhs, dimensions=['ydim', 'xdim', 'time'])

                parent = Field(variables=[tas, rhs])
            else:
                parent = None

        else:
            vx, vy, parent = [None] * 3

        svx = variable_scatter(vx, dest_mpi)
        svy = variable_scatter(vy, dest_mpi)

        if with_parent:
            parent = variable_collection_scatter(parent, dest_mpi)
            kwds['parent'] = parent
        kwds[KeywordArgument.ABSTRACTION] = abstraction

        grid = Grid(svx, svy, **kwds)

        return grid

    def get_gridxy_global(self, *args, **kwargs):
        return create_gridxy_global(*args, **kwargs)

    def get_geometryvariable(self, **kwargs):
        value_point_array = np.array([None, None])
        value_point_array[:] = [Point(1, 2), Point(3, 4)]
        if kwargs.get('value') is None:
            kwargs['value'] = kwargs.pop('value', value_point_array)
        if kwargs.get('dimensions') is None:
            kwargs['dimensions'] = 'ngeom'
        pa = GeometryVariable(**kwargs)
        return pa

    def get_request_dataset(self, **kwargs):
        data = self.test_data.get_rd('cancm4_tas', kwds=kwargs)
        return data

    def get_variable_x(self, bounds=True):
        value = [-100., -99., -98., -97.]
        if bounds:
            bounds = [[v - 0.5, v + 0.5] for v in value]
            bounds = Variable(value=bounds, name='x_bounds', dimensions=['x', 'bounds'])
        else:
            bounds = None
        x = Variable(value=value, bounds=bounds, name='x', dimensions='x')
        return x

    def get_variable_y(self, bounds=True):
        value = [40., 39., 38.]
        if bounds:
            bounds = [[v + 0.5, v - 0.5] for v in value]
            bounds = Variable(value=bounds, name='y_bounds', dimensions=['y', 'bounds'])
        else:
            bounds = None
        y = Variable(value=value, bounds=bounds, name='y', dimensions='y')
        return y

    @property
    def polygon_value(self):
        polys = [['POLYGON ((-100.5 40.5, -99.5 40.5, -99.5 39.5, -100.5 39.5, -100.5 40.5))',
                  'POLYGON ((-99.5 40.5, -98.5 40.5, -98.5 39.5, -99.5 39.5, -99.5 40.5))',
                  'POLYGON ((-98.5 40.5, -97.5 40.5, -97.5 39.5, -98.5 39.5, -98.5 40.5))',
                  'POLYGON ((-97.5 40.5, -96.5 40.5, -96.5 39.5, -97.5 39.5, -97.5 40.5))'],
                 ['POLYGON ((-100.5 39.5, -99.5 39.5, -99.5 38.5, -100.5 38.5, -100.5 39.5))',
                  'POLYGON ((-99.5 39.5, -98.5 39.5, -98.5 38.5, -99.5 38.5, -99.5 39.5))',
                  'POLYGON ((-98.5 39.5, -97.5 39.5, -97.5 38.5, -98.5 38.5, -98.5 39.5))',
                  'POLYGON ((-97.5 39.5, -96.5 39.5, -96.5 38.5, -97.5 38.5, -97.5 39.5))'],
                 ['POLYGON ((-100.5 38.5, -99.5 38.5, -99.5 37.5, -100.5 37.5, -100.5 38.5))',
                  'POLYGON ((-99.5 38.5, -98.5 38.5, -98.5 37.5, -99.5 37.5, -99.5 38.5))',
                  'POLYGON ((-98.5 38.5, -97.5 38.5, -97.5 37.5, -98.5 37.5, -98.5 38.5))',
                  'POLYGON ((-97.5 38.5, -96.5 38.5, -96.5 37.5, -97.5 37.5, -97.5 38.5))']]
        return self.get_shapely_from_wkt_array(polys)

    def get_polygonarray(self):
        grid = self.get_polygon_array_grid()
        poly = get_geometry_variable(grid)
        return poly

    def get_polygon_array_grid(self, with_bounds=True):
        if with_bounds:
            xb = Variable(value=[[-100.5, -99.5], [-99.5, -98.5], [-98.5, -97.5], [-97.5, -96.5]], name='xb',
                          dimensions=['x', 'bounds'])
            yb = Variable(value=[[40.5, 39.5], [39.5, 38.5], [38.5, 37.5]], name='yb', dimensions=['y', 'bounds'])
        else:
            xb, yb = [None, None]
        x = Variable(value=[-100.0, -99.0, -98.0, -97.0], bounds=xb, name='x', dimensions='x')
        y = Variable(value=[40.0, 39.0, 38.0], name='y', bounds=yb, dimensions='y')
        grid = Grid(x, y)
        return grid

    def get_shapely_from_wkt_array(self, wkts):
        ret = np.array(wkts)
        vfunc = np.vectorize(wkt.loads, otypes=[object])
        ret = vfunc(ret)
        ret = np.ma.array(ret, mask=False)
        return ret

    @staticmethod
    def write_fiona_htmp(obj, name):
        path = os.path.join('/home/benkoziol/htmp/ocgis', '{}.shp'.format(name))
        obj.write_fiona(path)


class AbstractTestField(TestBase):
    def setUp(self):
        np.random.seed(1)
        super(AbstractTestField, self).setUp()

    def get_col(self, bounds=True, with_name=True):
        value = [-100., -99., -98., -97.]
        if bounds:
            bounds = [[v - 0.5, v + 0.5] for v in value]
            bounds_name = 'longitude_bounds' if with_name else None
            bounds = Variable(value=bounds, name=bounds_name, dimensions=['lon', 'bounds'])
        else:
            bounds = None
        name = 'longitude' if with_name else None
        col = Variable(value=value, bounds=bounds, name=name, dimensions=['lon'])
        return col

    def get_row(self, bounds=True, with_name=True):
        value = [40., 39., 38.]
        if bounds:
            bounds = [[v + 0.5, v - 0.5] for v in value]
            bounds_name = 'latitude_bounds' if with_name else None
            bounds = Variable(value=bounds, name=bounds_name, dimensions=['lat', 'bounds'])
        else:
            bounds = None
        name = 'latitude' if with_name else None
        row = Variable(value=value, bounds=bounds, name=name, dimensions='lat')
        return row

    def get_field(self, with_bounds=True, with_value=False, with_level=True, with_temporal=True, with_realization=True,
                  month_count=1, name='tmax', units='kelvin', field_name=None, crs=None, with_dimension_names=True):
        from datetime import datetime as dt

        if with_temporal:
            temporal_start = dt(2000, 1, 1, 12)
            if month_count == 1:
                temporal_stop = dt(2000, 1, 31, 12)
            elif month_count == 2:
                temporal_stop = dt(2000, 2, 29, 12)
            else:
                temporal_stop = dt(2000 + month_count // 12,
                                   1 + month_count % 12, 1, 12)
            temporal_value = get_date_list(temporal_start, temporal_stop, 1)
            delta_bounds = datetime.timedelta(hours=12)
            if with_bounds:
                temporal_bounds = [[v - delta_bounds, v + delta_bounds] for v in temporal_value]
                time_bounds_name = 'time_bnds' if with_dimension_names else None
                temporal_bounds = TemporalVariable(value=temporal_bounds, name=time_bounds_name,
                                                   dimensions=['time', 'bounds'])
            else:
                temporal_bounds = None
            dname = 'time' if with_dimension_names else None
            temporal = TemporalVariable(value=temporal_value, bounds=temporal_bounds, name=dname, dimensions='time')
            t_shape = temporal.shape[0]
        else:
            temporal = None
            t_shape = 1

        if with_level:
            level_value = [50, 150]
            if with_bounds:
                level_bounds_name = 'level_bnds' if with_dimension_names else None
                level_bounds = [[0, 100], [100, 200]]
                level_bounds = Variable(value=level_bounds, name=level_bounds_name, units='meters',
                                        dimensions=['level', 'bounds'])
            else:
                level_bounds = None
            dname = 'level' if with_dimension_names else None
            level = Variable(value=level_value, bounds=level_bounds, name=dname, units='meters', dimensions='level')
            l_shape = level.shape[0]
        else:
            level = None
            l_shape = 1

        with_name = True if with_dimension_names else False
        row = self.get_row(bounds=with_bounds, with_name=with_name)
        col = self.get_col(bounds=with_bounds, with_name=with_name)
        grid = Grid(col, row, crs=crs)
        row_shape = row.shape[0]
        col_shape = col.shape[0]

        if with_realization:
            dname = 'realization' if with_dimension_names else None
            realization = Variable(value=[1, 2], name=dname, dimensions='realization')
            r_shape = realization.shape[0]
        else:
            realization = None
            r_shape = 1

        if with_value:
            value = np.random.rand(r_shape, t_shape, l_shape, row_shape, col_shape)
            data = None
        else:
            value = None
            data = 'foo'

        var = SourcedVariable(name, units=units, request_dataset=data, value=value,
                              dimensions=['realization', 'time', 'level', 'lat', 'lon'])
        field = Field(variables=var, time=temporal, level=level, realization=realization, grid=grid,
                      name=field_name, is_data=name)

        return field


def create_exact_field(grid, data_varname, ntime=1, fill_data_var=True, crs='auto'):
    tdim = Dimension(name='time', size=None, size_current=ntime)
    tvar = TemporalVariable(name='time', value=range(1, ntime + 1), dimensions=tdim, dtype=np.float32,
                            attrs={'axis': 'T'})
    dvar_dims = [tdim] + list(grid.dimensions)
    dvar = Variable(name=data_varname, dimensions=dvar_dims, dtype=np.float32)
    if fill_data_var:
        if grid.is_vectorized:
            longitude, latitude = np.meshgrid(grid.x.get_value(), grid.y.get_value())
        else:
            longitude, latitude = grid.x.get_value(), grid.y.get_value()
        exact = create_exact_field_value(longitude, latitude)
        to_fill = dvar.get_value()
        to_fill[:, :, :] = exact
        for tidx in range(ntime):
            to_fill[tidx, :, :] = to_fill[tidx, :, :] + ((tidx + 1) * 10)

    field = Field(grid=grid, time=tvar, is_data=dvar, crs=crs)

    return field


def create_gridxy_global(resolution=1.0, with_bounds=True, wrapped=True, crs=None, dtype=None, dist=True):
    half_resolution = 0.5 * resolution
    y = np.arange(-90.0 + half_resolution, 90.0, resolution)
    if wrapped:
        x = np.arange(-180.0 + half_resolution, 180.0, resolution)
    else:
        x = np.arange(0.0 + half_resolution, 360.0, resolution)

    if dist:
        ompi = OcgDist()
        ompi.create_dimension('x', x.shape[0], dist=False)
        ompi.create_dimension('y', y.shape[0], dist=True)
        ompi.update_dimension_bounds()

    if MPI_RANK == 0:
        x = Variable(name='x', value=x, dimensions='x', dtype=dtype)
        y = Variable(name='y', value=y, dimensions='y', dtype=dtype)
    else:
        x, y = [None] * 2

    if dist:
        x = variable_scatter(x, ompi)
        y = variable_scatter(y, ompi)

    grid = Grid(x, y, crs=crs)

    if with_bounds:
        grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')

    return grid


def get_geometry_dictionaries(uid='UGID'):
    coordinates = [('France', [2.8, 47.16]),
                   ('Germany', [10.5, 51.29]),
                   ('Italy', [12.2, 43.4])]
    geom = []
    for ugid, coordinate in enumerate(coordinates, start=1):
        point = Point(coordinate[1][0], coordinate[1][1])
        geom.append({'geom': point,
                     'properties': {uid: ugid, 'COUNTRY': coordinate[0]}})
    return geom
