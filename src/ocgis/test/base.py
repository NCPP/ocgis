import unittest
import abc
import tempfile
from ocgis import env
import shutil
from copy import deepcopy, copy
import os
from collections import OrderedDict
import subprocess
import ocgis
from warnings import warn
from subprocess import CalledProcessError
import numpy as np
from ocgis.api.request.base import RequestDataset
import netCDF4 as nc
from ocgis.util.helpers import get_iter


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

        base_dir = os.path.split(__file__)[0]
        ret = os.path.join(base_dir, 'bin')
        return ret

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
                    raise KeyError(msg)
                self.assertEqual(v, d2[k], msg=msg)
            self.assertEqual(set(d1.keys()), set(d2.keys()))

    def assertNumpyAll(self, arr1, arr2, check_fill_value_dtype=True, check_arr_dtype=True):
        """
        Asserts arrays are equal according to the test criteria.

        :param arr1: An array to compare.
        :type arr1: :class:`numpy.ndarray`
        :param arr2: An array to compare.
        :type arr2: :class:`numpy.ndarray`
        :param bool check_fill_value_dtype: If ``True``, check that the data type for masked array fill values are equal.
        :param bool check_arr_dtype: If ``True``, check the data types of the arrays are equal.
        :raises: AssertionError
        """

        self.assertEqual(type(arr1), type(arr2))
        self.assertEqual(arr1.shape, arr2.shape)
        if check_arr_dtype:
            self.assertEqual(arr1.dtype, arr2.dtype)
        if isinstance(arr1, np.ma.MaskedArray) or isinstance(arr2, np.ma.MaskedArray):
            self.assertTrue(np.all(arr1.data == arr2.data))
            self.assertTrue(np.all(arr1.mask == arr2.mask))
            if check_fill_value_dtype:
                self.assertEqual(arr1.fill_value, arr2.fill_value)
            else:
                self.assertTrue(np.equal(arr1.fill_value, arr2.fill_value.astype(arr1.fill_value.dtype)))
        else:
            self.assertTrue(np.all(arr1 == arr2))

    def assertNcEqual(self, uri_src, uri_dest, check_types=True, close=False, metadata_only=False,
                      ignore_attributes=None):
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

        :raises: AssertionError
        """

        src = nc.Dataset(uri_src)
        dest = nc.Dataset(uri_dest)

        ignore_attributes = ignore_attributes or {}

        try:
            for dimname, dim in src.dimensions.iteritems():
                self.assertEqual(len(dim), len(dest.dimensions[dimname]))
            self.assertEqual(set(src.dimensions.keys()), set(dest.dimensions.keys()))

            for varname, var in src.variables.iteritems():
                dvar = dest.variables[varname]
                try:
                    if not metadata_only:
                        if close:
                            self.assertNumpyAllClose(var[:], dvar[:])
                        else:
                            self.assertNumpyAll(var[:], dvar[:], check_arr_dtype=check_types)
                except AssertionError:
                    cmp = var[:] == dvar[:]
                    if cmp.shape == (1,) and cmp.data[0] == True:
                        pass
                    else:
                        raise
                if check_types:
                    self.assertEqual(var[:].dtype, dvar[:].dtype)

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
                        msg = 'The attribute "{0}" is not found on the variable "{1}" for URI "{2}".'\
                            .format(k, dvar._name, uri_dest)
                        raise AttributeError(msg)
                    try:
                        self.assertNumpyAll(v, to_test_attr)
                    except AttributeError:
                        self.assertEqual(v, to_test_attr)
                self.assertEqual(var.dimensions, dvar.dimensions)
            self.assertEqual(set(src.variables.keys()), set(dest.variables.keys()))

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

    def get_temporary_output_directory(self):
        """
        :returns: A path to a temporary directory with an appropriate prefix.
        :rtype: str
        """

        return tempfile.mkdtemp(prefix=self._prefix_path_test)

    @staticmethod
    def get_tst_data():
        """
        :returns: A dictionary-like object with special access methods for test files.
        :rtype: :class:`ocgis.test.base.TestData`
        """

        test_data = TestData()

        test_data.update(['nc', 'CMIP3'], 'Tavg', 'Extraction_Tavg.nc', key='cmip3_extraction')
        test_data.update(['nc', 'CanCM4'], 'rhs', 'rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc', key='cancm4_rhs')
        test_data.update(['nc', 'CanCM4'], 'rhsmax', 'rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc', key='cancm4_rhsmax')
        test_data.update(['nc', 'CanCM4'], 'tas', 'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc', key='cancm4_tas')
        test_data.update(['nc', 'CanCM4'], 'tasmax', 'tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc', key='cancm4_tasmax_2001')
        test_data.update(['nc', 'CanCM4'], 'tasmax', 'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc', key='cancm4_tasmax_2011')
        test_data.update(['nc', 'CanCM4'], 'tasmin', 'tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc', key='cancm4_tasmin_2001')
        test_data.update(['nc', 'daymet'], 'tmax', 'tmax.nc', key='daymet_tmax')
        test_data.update(['nc', 'maurer', '2010'], 'pr', ['nldas_met_update.obs.daily.pr.1990.nc', 'nldas_met_update.obs.daily.pr.1991.nc'], key='maurer_2010_pr')
        test_data.update(['nc', 'maurer', '2010'], 'tas', ['nldas_met_update.obs.daily.tas.1990.nc', 'nldas_met_update.obs.daily.tas.1991.nc'], key='maurer_2010_tas')
        test_data.update(['nc', 'maurer', '2010'], 'tasmax', ['nldas_met_update.obs.daily.tasmax.1990.nc', 'nldas_met_update.obs.daily.tasmax.1991.nc'], key='maurer_2010_tasmax')
        test_data.update(['nc', 'maurer', '2010'], 'tasmin', ['nldas_met_update.obs.daily.tasmin.1990.nc', 'nldas_met_update.obs.daily.tasmin.1991.nc'], key='maurer_2010_tasmin')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'tasmax', 'Maurer02new_OBS_tasmax_daily.1971-2000.nc', key='maurer_2010_concatenated_tasmax')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'tasmin', 'Maurer02new_OBS_tasmin_daily.1971-2000.nc', key='maurer_2010_concatenated_tasmin')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'tas', 'Maurer02new_OBS_tas_daily.1971-2000.nc', key='maurer_2010_concatenated_tas')
        test_data.update(['nc', 'maurer', '2010-concatenated'], 'pr', 'Maurer02new_OBS_pr_daily.1971-2000.nc', key='maurer_2010_concatenated_pr')
        test_data.update(['nc', 'maurer', 'bcca'], 'tasmax', 'gridded_obs.tasmax.OBS_125deg.daily.1991.nc', key='maurer_bcca_1991')
        test_data.update(['nc', 'maurer', 'bccr'], 'Prcp', 'bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc', key='maurer_bccr_1950')
        test_data.update(['nc', 'misc', 'month_in_time_units'], 'clt', 'clt.nc', key='clt_month_units')
        test_data.update(['nc', 'misc', 'rotated_pole'], 'pr', 'pr_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_mon_198101-199012.nc', key='rotated_pole_cnrm_cerfacs')
        test_data.update(['nc', 'misc', 'rotated_pole'], 'tas', 'tas_EUR-44_CCCma-CanESM2_rcp85_r1i1p1_SMHI-RCA4_v1_sem_209012-210011.nc', key='rotated_pole_cccma')
        test_data.update(['nc', 'misc', 'rotated_pole'], 'tas', 'tas_EUR-44_ICHEC-EC-EARTH_historical_r12i1p1_SMHI-RCA4_v1_day_19710101-19751231.nc', key='rotated_pole_ichec')
        test_data.update(['nc', 'misc', 'subset_test'], 'Prcp', 'sresa2.ncar_pcm1.3.monthly.Prcp.RAW.1950-2099.nc', key='subset_test_Prcp')
        test_data.update(['nc', 'misc', 'subset_test'], 'Tavg', 'Tavg_bccr_bcm2_0.1.sresa2.nc', key='subset_test_Tavg')
        test_data.update(['nc', 'misc', 'subset_test'], 'Tavg', 'sresa2.bccr_bcm2_0.1.monthly.Tavg.RAW.1950-2099.nc', key='subset_test_Tavg_sresa2')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_CRCM_ccsm_1981010103.nc', key='narccap_crcm')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_CRCM_ccsm_1981010103.nc', key='narccap_polar_stereographic')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_HRM3_gfdl_1981010103.nc', key='narccap_hrm3')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_RCM3_gfdl_1981010103.nc', key='narccap_rcm3')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_WRFG_ccsm_1986010103.nc', key='narccap_lambert_conformal')
        test_data.update(['nc', 'narccap'], 'pr', 'pr_WRFG_ccsm_1986010103.nc', key='narccap_wrfg')
        test_data.update(['nc', 'narccap'], 'pr', ['pr_WRFG_ncep_1981010103.nc', 'pr_WRFG_ncep_1986010103.nc'], key='narccap_pr_wrfg_ncep')
        test_data.update(['nc', 'narccap'], 'tas', 'tas_HRM3_gfdl_1981010103.nc', key='narccap_rotated_pole')
        test_data.update(['nc', 'narccap'], 'tas', 'tas_RCM3_gfdl_1981010103.nc', key='narccap_tas_rcm3_gfdl')
        test_data.update(['nc', 'QED-2013'], 'climatology_TNn_monthly_max', 'climatology_TNn_monthly_max.nc', key='qed_2013_TNn_monthly_max')
        test_data.update(['nc', 'QED-2013'], 'climatology_TNn_annual_min', 'climatology_TNn_annual_min.nc', key='qed_2013_TNn_annual_min')
        test_data.update(['nc', 'QED-2013'], 'climatology_TasMin_seasonal_max_of_seasonal_means', 'climatology_TasMin_seasonal_max_of_seasonal_means.nc', key='qed_2013_TasMin_seasonal_max_of_seasonal_means')
        test_data.update(['nc', 'QED-2013'], 'climatology_Tas_annual_max_of_annual_means', 'climatology_Tas_annual_max_of_annual_means.nc', key='qed_2013_climatology_Tas_annual_max_of_annual_means')
        test_data.update(['nc', 'QED-2013', 'multifile'], 'txxmmedm', 'maurer02v2_median_txxmmedm_january_1971-2000.nc', key='qed_2013_maurer02v2_median_txxmmedm_january_1971-2000')
        test_data.update(['nc', 'QED-2013', 'multifile'], 'txxmmedm', 'maurer02v2_median_txxmmedm_february_1971-2000.nc', key='qed_2013_maurer02v2_median_txxmmedm_february_1971-2000')
        test_data.update(['nc', 'QED-2013', 'multifile'], 'txxmmedm', 'maurer02v2_median_txxmmedm_march_1971-2000.nc', key='qed_2013_maurer02v2_median_txxmmedm_march_1971-2000')
        test_data.update(['nc', 'snippets'], 'dtr', 'snippet_Maurer02new_OBS_dtr_daily.1971-2000.nc', key='snippet_maurer_dtr')
        test_data.update(['nc', 'snippets'], 'bias', 'seasonalbias.nc', key='snippet_seasonalbias')

        return test_data

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
