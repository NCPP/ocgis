import os
import tempfile
from importlib import import_module

import numpy as np
from ocgis import constants
from ocgis import env, OcgOperations
from ocgis.environment import EnvParmImport
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class TestEnvImportParm(TestBase):
    reset_env = False

    def test_constructor(self):
        pm = EnvParmImport('USE_NUMPY', None, 'numpy')
        self.assertEqual(pm.value, True)
        self.assertEqual(pm.module_names, 'numpy')

    def test_bad_import(self):
        pm = EnvParmImport('USE_FOO', None, 'foo')
        self.assertEqual(pm.value, False)

    def test_import_available_overloaded(self):
        pm = EnvParmImport('USE_NUMPY', False, 'numpy')
        self.assertEqual(pm.value, False)

    def test_environment_variable(self):
        os.environ['OCGIS_USE_FOOL_GOLD'] = 'True'
        pm = EnvParmImport('USE_FOOL_GOLD', None, 'foo')
        self.assertEqual(pm.value, True)


class TestEnvironment(TestBase):
    reset_env = False

    def get_is_available(self, module_name):
        try:
            import_module(module_name)
            av = True
        except ImportError:
            av = False
        return av

    def test_init(self):
        self.assertIsNone(env.MELTED)
        self.assertEqual(env.DEFAULT_GEOM_UID, constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER)
        self.assertEqual(env.NETCDF_FILE_FORMAT, constants.NETCDF_DEFAULT_DATA_MODEL)
        self.assertEqual(env.NP_INT, env.NP_INT)
        self.assertEqual(env.NP_FLOAT, env.NP_FLOAT)

        self.assertIsInstance(env.USE_NETCDF4_MPI, bool)
        env.reset()
        self.assertIsInstance(env.USE_NETCDF4_MPI, bool)

    def test_conf_path(self):
        env.CONF_PATH

    def test_default_coordsys(self):
        env.DEFAULT_COORDSYS

    @attr('data')
    def test_env_overload(self):
        # check env overload
        out = tempfile.mkdtemp()
        try:
            env.DIR_OUTPUT = out
            env.PREFIX = 'my_prefix'
            rd = self.test_data.get_rd('daymet_tmax')
            ops = OcgOperations(dataset=rd, snippet=True)
            self.assertEqual(env.DIR_OUTPUT, ops.dir_output)
            self.assertEqual(env.PREFIX, ops.prefix)
        finally:
            os.rmdir(out)
            env.reset()

    def test_import_attributes(self):
        try:
            # With both modules installed, these are expected to be true.
            self.assertEqual(env.USE_CFUNITS, self.get_is_available('cf_units'))
        except AssertionError:
            # Try the other unit conversion library.
            self.assertEqual(env.USE_CFUNITS, self.get_is_available('cfunits'))

        self.assertEqual(env.USE_SPATIAL_INDEX, self.get_is_available('rtree'))

        # Turn off the spatial index.
        env.USE_SPATIAL_INDEX = False
        self.assertEqual(env.USE_SPATIAL_INDEX, False)
        env.reset()
        self.assertEqual(env.USE_SPATIAL_INDEX, self.get_is_available('rtree'))

    def test_import_attributes_overloaded(self):
        try:
            import rtree

            av = True
        except ImportError:
            av = False
        self.assertEqual(env.USE_SPATIAL_INDEX, av)

        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 'False'
        env.reset()
        self.assertFalse(env.USE_SPATIAL_INDEX)

        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 't'
        env.reset()
        self.assertTrue(env.USE_SPATIAL_INDEX)

        # this cannot be transformed into a boolean value, and it is also not a realistic module name, so it will
        # evaluate to false
        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 'False'
        env.reset()
        self.assertFalse(env.USE_SPATIAL_INDEX)

        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 'f'
        env.reset()
        self.assertFalse(env.USE_SPATIAL_INDEX)

    def test_netcdf_file_format(self):
        try:
            self.assertEqual(env.NETCDF_FILE_FORMAT, constants.NETCDF_DEFAULT_DATA_MODEL)
            self.assertEqual(env.NP_INT, env.NP_INT)
            actual = 'NETCDF3_CLASSIC'
            env.NETCDF_FILE_FORMAT = actual
        finally:
            env.reset()

    def test_np_float(self):
        try:
            self.assertEqual(env.NP_FLOAT, env.NP_FLOAT)
            env.NP_FLOAT = np.int16
            self.assertEqual(env.NP_FLOAT, np.int16)
        finally:
            env.reset()
            self.assertEqual(env.NP_FLOAT, constants.DEFAULT_NP_FLOAT)

    def test_np_int(self):
        try:
            self.assertEqual(env.NP_INT, env.NP_INT)
            env.NP_INT = np.int16
            self.assertEqual(env.NP_INT, np.int16)
        finally:
            env.reset()

    def test_reset(self):
        # Test netCDF MPI is reset appropriately.
        env.USE_NETCDF4_MPI = True
        self.assertTrue(env.USE_NETCDF4_MPI)
        env.reset()
        self.assertFalse(env.USE_NETCDF4_MPI)

    def test_simple(self):
        self.assertEqual(env.OVERWRITE, False)
        env.reset()
        os.environ['OCGIS_OVERWRITE'] = 't'
        self.assertEqual(env.OVERWRITE, True)
        env.OVERWRITE = False
        self.assertEqual(env.OVERWRITE, False)
        with self.assertRaises(AttributeError):
            env.FOO = 1

        env.OVERWRITE = True
        self.assertEqual(env.OVERWRITE, True)
        env.reset()
        os.environ.pop('OCGIS_OVERWRITE')
        self.assertEqual(env.OVERWRITE, False)

    def test_str(self):
        ret = str(env)
        self.assertTrue(len(ret) > 300)

    def test_suppress_warnings(self):
        self.assertTrue(env.SUPPRESS_WARNINGS)
