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

import fiona
import netCDF4 as nc
import numpy as np

import ocgis
from ocgis import env
from ocgis.api.collection import SpatialCollection
from ocgis.api.request.base import RequestDataset
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, SpatialDimension
from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.interface.base.field import Field
from ocgis.interface.base.variable import Variable
from ocgis.util.geom_cabinet import GeomCabinet
from ocgis.util.helpers import get_iter
from ocgis.util.itester import itr_products_keywords
from ocgis.util.logging_ocgis import ocgis_lh

"""
Definitions for various "attrs":
 * slow: long-running tests that are typically ran before a release
 * remote: tests relying on remote datasets that are typically run before a release
 * esmf: tests requiring ESMF
 * simple: simple tests designed to test core functionality requiring no datasets
 * optional: tests that use optional dependencies
 * data: test requires a data file
 * icclim: test requires ICCLIM

nosetests -vs --with-id -a '!slow,!remote' ocgis
"""


class ToTest(Exception):
    """
    Useful when wanting to flag things as not tested.
    """


class TestBase(unittest.TestCase):
    """
    All tests should inherit from this. It allows test data to be written to a temporary folder and removed easily.
    Also simplifies access to static files.
    """

    __metaclass__ = abc.ABCMeta
    # set to false to not resent the environment before each test
    reset_env = True
    # set to false to not create and destroy a temporary directory before each test
    create_dir = True
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
        actual_descriptives = self.get_descriptive_statistics(actual_arr)
        for k, v in desired_descriptives.items():
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
            for k, v in d1.iteritems():
                try:
                    msg = 'Issue with key "{0}". Values are {1}.'.format(k, (v, d2[k]))
                except KeyError:
                    msg = 'The key "{0}" was not found in the second dictionary.'.format(k)
                    raise AssertionError(msg)
                self.assertEqual(v, d2[k], msg=msg)
            self.assertEqual(set(d1.keys()), set(d2.keys()))

    def assertFionaMetaEqual(self, meta, actual, abs_dtype=True):
        self.assertEqual(meta['crs'], actual['crs'])
        self.assertEqual(meta['driver'], actual['driver'])

        schema_meta = meta['schema']
        schema_actual = actual['schema']
        self.assertEqual(schema_meta['geometry'], schema_actual['geometry'])

        properties_meta = schema_meta['properties']
        properties_actual = schema_actual['properties']

        for km in properties_meta.iterkeys():
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

    def assertNumpyAll(self, arr1, arr2, check_fill_value_dtype=True, check_arr_dtype=True, check_arr_type=True,
                       rtol=None):
        """
        Asserts arrays are equal according to the test criteria.

        :param arr1: An array to compare.
        :type arr1: :class:`numpy.ndarray`
        :param arr2: An array to compare.
        :type arr2: :class:`numpy.ndarray`
        :param bool check_fill_value_dtype: If ``True``, check that the data type for masked array fill values are equal.
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
            data_to_check = (arr1.data, arr2.data)
            self.assertTrue(np.all(arr1.mask == arr2.mask))
            if check_fill_value_dtype:
                self.assertEqual(arr1.fill_value, arr2.fill_value)
            else:
                self.assertTrue(np.equal(arr1.fill_value, arr2.fill_value.astype(arr1.fill_value.dtype)))
        else:
            data_to_check = (arr1, arr2)

        # Check the data values.
        if rtol is None:
            to_assert = np.all(data_to_check[0] == data_to_check[1])
        else:
            to_assert = np.allclose(data_to_check[0], data_to_check[1], rtol=rtol)
        self.assertTrue(to_assert)

    def assertNcEqual(self, uri_src, uri_dest, check_types=True, close=False, metadata_only=False,
                      ignore_attributes=None, ignore_variables=None):
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
            for dimname, dim in src.dimensions.iteritems():
                self.assertEqual(len(dim), len(dest.dimensions[dimname]))
            self.assertEqual(set(src.dimensions.keys()), set(dest.dimensions.keys()))

            for varname, var in src.variables.iteritems():

                if varname in ignore_variables:
                    continue

                dvar = dest.variables[varname]

                var_value = var[:]
                dvar_value = dvar[:]

                try:
                    if not metadata_only:
                        if close:
                            self.assertNumpyAllClose(var_value, dvar_value)
                        else:
                            self.assertNumpyAll(var_value, dvar_value, check_arr_dtype=check_types)
                except AssertionError:
                    cmp = var_value == dvar_value
                    if cmp.shape == (1,) and cmp.data[0] == True:
                        pass
                    else:
                        raise

                if check_types:
                    self.assertEqual(var_value.dtype, dvar_value.dtype)

                # check values of attributes on all variables
                for k, v in var.__dict__.iteritems():
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
                for k, v in dvar.__dict__.iteritems():
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
                for k, v in src.__dict__.iteritems():
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

    def assertWarns(self, warning, meth):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            meth()
            self.assertTrue(any(item.category == warning for item in warning_list))

    def get_descriptive_statistics(self, arr):
        ret = {'mean': arr.mean(),
               'min': arr.min(),
               'max': arr.max(),
               'std': np.std(arr),
               'shape': arr.shape}
        if arr.ndim == 1:
            arr = np.diagflat(arr)
        ret['trace'] = np.trace(arr)
        return ret

    def get_esmf_field(self, **kwargs):
        """
        :keyword field: (``=None``) The field object. If ``None``, call :meth:`~ocgis.test.base.TestBase.get_field`
        :type field: :class:`~ocgis.Field`
        :param kwargs: Other keyword arguments to :meth:`ocgis.test.base.TestBase.get_field`.
        :returns: An ESMF field object.
        :rtype: :class:`ESMF.Field`
        """

        from ocgis.conv.esmpy import ESMPyConverter

        field = kwargs.pop('field', None) or self.get_field(**kwargs)
        coll = SpatialCollection()
        coll.add_field(field)
        conv = ESMPyConverter([coll])
        efield = conv.write()
        return efield

    def get_field(self, nlevel=None, nrlz=None, crs=None, ntime=2, with_bounds=False):
        """
        :param int nlevel: The number of level elements.
        :param int nrlz: The number of realization elements.
        :param crs: The coordinate system for the field.
        :type crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :param ntime: The number of time elements.
        :type ntime: int
        :param with_bounds: If ``True``, extrapolate bounds on spatial dimensions.
        :type with_bounds: bool
        :returns: A small field object for testing.
        :rtype: `~ocgis.Field`
        """

        np.random.seed(1)
        row = VectorDimension(value=[4., 5.], name='row')
        col = VectorDimension(value=[40., 50.], name='col')

        if with_bounds:
            row.set_extrapolated_bounds()
            col.set_extrapolated_bounds()

        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid, crs=crs)

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
        temporal = TemporalDimension(value=value_temporal)

        if nlevel is None:
            nlevel = 1
            level = None
        else:
            level = VectorDimension(value=range(1, nlevel + 1), name='level')

        if nrlz is None:
            nrlz = 1
            realization = None
        else:
            realization = VectorDimension(value=range(1, nrlz + 1), name='realization')

        variable = Variable(name='foo', value=np.random.rand(nrlz, ntime, nlevel, 2, 2))
        field = Field(spatial=sdim, temporal=temporal, variables=variable, level=level, realization=realization)

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

        return tempfile.mkdtemp(prefix=self._prefix_path_test)

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

    def inspect(self, uri, variable=None):
        from ocgis.util.inspect import Inspect

        print Inspect(uri, variable=None)

    def iter_product_keywords(self, keywords, as_namedtuple=True):
        return itr_products_keywords(keywords, as_namedtuple=as_namedtuple)

    def nautilus(self, path):
        if not os.path.isdir(path):
            path = os.path.split(path)[0]
        subprocess.call(['nautilus', path])

    def nc_scope(self, *args, **kwargs):
        return nc_scope(*args, **kwargs)

    def panoply(self, *args):
        paths = args[0]
        if isinstance(paths, basestring):
            paths = [paths]
        paths = list(paths)
        cmd = ['/home/benkoziol/sandbox/PanoplyJ/panoply.sh'] + paths
        subprocess.check_call(cmd)

    def print_scope(self):
        return print_scope()

    def setUp(self):
        self.current_dir_output = None
        if self.reset_env:
            env.reset()
        if self.create_dir:
            self.current_dir_output = self.get_temporary_output_directory()
            env.DIR_OUTPUT = self.current_dir_output

    def shortDescription(self):
        """
        Overloaded method so ``nose`` will not print the docstring associated with a test.
        """

        return None

    def tearDown(self):
        try:
            if self.create_dir:
                shutil.rmtree(self.current_dir_output)
        finally:
            if self.reset_env:
                env.reset()
            if self.shutdown_logging:
                ocgis_lh.shutdown()


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
        for key in self.keys():
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
        if isinstance(filename, basestring):
            coll.append(filename)
            uri = os.path.join(*coll)
        else:
            uri = []
            for part in filename:
                copy_coll = copy(coll)
                copy_coll.append(part)
                uri.append(os.path.join(*copy_coll))

        # ensure the uris exist, if not, we may need to download
        if isinstance(uri, basestring):
            assert (os.path.exists(uri))
        else:
            for element in uri:
                assert (os.path.exists(element))

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

    http://nose.readthedocs.org/en/latest/plugins/attrib.html
    """

    def wrap_ob(ob):
        for name in args:
            setattr(ob, name, True)
        for name, value in kwargs.iteritems():
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