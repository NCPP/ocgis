import os

import numpy as np
from ocgis.driver.operations import OcgOperations

import ocgis
from ocgis.driver.request import RequestDataset
from ocgis.exc import DefinitionValidationError, ExtentError
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.variable.crs import CFRotatedPole, CFWGS84


class TestRotatedPole(TestBase):
    @attr('data')
    def test_validation(self):
        # CFRotatedPole is not an appropriate output crs. it may also not be transformed to anything but WGS84
        rd = self.test_data.get_rd('narccap_rotated_pole')
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, output_crs=CFRotatedPole(grid_north_pole_latitude=5,
                                                               grid_north_pole_longitude=5))
        # this is an okay output coordinate system for the two input coordinate systems
        rd2 = self.test_data.get_rd('narccap_lambert_conformal')
        OcgOperations(dataset=[rd, rd2], output_crs=CFWGS84())

    @attr('data')
    def test_calculation(self):
        rd = self.test_data.get_rd('narccap_rotated_pole', kwds=dict(time_region={'month': [12], 'year': [1982]}))
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            output_format='nc')
        ret = ops.execute()
        field = ocgis.RequestDataset(uri=ret, variable='mean').get()
        self.assertIsInstance(field.spatial.crs, CFRotatedPole)
        self.assertEqual(field.shape, (1, 1, 1, 130, 155))

    @attr('data')
    def test_intersects(self):
        rd = self.test_data.get_rd('narccap_rotated_pole', kwds=dict(time_region={'month': [12], 'year': [1982]}))
        ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[16])
        ret = ops.execute()
        ref = ret.gvu(16, 'tas')
        self.assertEqual(ref.shape, (1, 248, 1, 7, 15))
        self.assertAlmostEqual(np.ma.mean(ref), 269.8304976396538)
        self.assertNumpyAll(ref.mask.squeeze()[0, :, :],
                            np.array([[True, True, True, True, False, False, False, False, False, False, False, False,
                                       False, False, False],
                                      [True, True, True, True, False, False, False, False, False, False, False, False,
                                       False, False, False],
                                      [True, True, True, True, False, False, False, False, False, False, False, False,
                                       False, False, True],
                                      [False, False, False, False, False, False, False, False, False, False, False,
                                       False, False, False, True],
                                      [False, False, False, False, False, False, False, False, False, False, False,
                                       False, False, False, True],
                                      [True, False, False, False, False, False, False, False, False, False, False,
                                       False, False, False, True],
                                      [True, False, False, False, False, False, False, False, False, False, True, True,
                                       True, True, True]], dtype=bool))

    @attr('data')
    def test_clip_aggregate(self):
        rd = self.test_data.get_rd('narccap_rotated_pole', kwds=dict(time_region={'month': [12], 'year': [1982]}))
        ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[16],
                            spatial_operation='clip', aggregate=True, output_format='numpy')
        # the output CRS should be automatically updated for this operation
        self.assertEqual(ops.output_crs, CFWGS84())
        ret = ops.execute()
        ret = ret.gvu(16, 'tas')
        self.assertEqual(ret.shape, (1, 248, 1, 1, 1))
        self.assertAlmostEqual(ret.mean(), 269.83051915322579)

    @attr('data')
    def test_read(self):
        rd = self.test_data.get_rd('narccap_rotated_pole')
        field = rd.get()
        self.assertIsInstance(field.spatial.crs, CFRotatedPole)

    @attr('data')
    def test_to_netcdf(self):
        rd = self.test_data.get_rd('narccap_rotated_pole', kwds=dict(time_region={'month': [12], 'year': [1982]}))
        # it does not care about slices or no geometries
        ops = OcgOperations(dataset=rd, output_format='nc')
        ret = ops.execute()
        rd2 = ocgis.RequestDataset(uri=ret, variable='tas')
        self.assertEqual(rd2.get().temporal.extent, (5444.0, 5474.875))

    @attr('data')
    def test_to_netcdf_with_geometry(self):
        rd = self.test_data.get_rd('narccap_rotated_pole')
        # this bounding box covers the entire spatial domain. the software will move between rotated pole and CFWGS84
        # using this operation. it can then be compared against the "null" result which just does a snippet.
        geom = [-173.3, 8.8, -20.6, 79.0]
        ops = OcgOperations(dataset=rd, output_format='nc', snippet=True, geom=geom)
        ret = ops.execute()
        ops2 = OcgOperations(dataset=rd, output_format='nc', snippet=True, prefix='hi')
        ret2 = ops2.execute()
        self.assertNcEqual(ret, ret2, metadata_only=True, ignore_attributes={'global': ['history']})

        with nc_scope(ret) as ds:
            with nc_scope(ret2) as ds2:
                for var_name in ['yc', 'xc', 'tas']:
                    var = ds.variables[var_name][:]
                    var2 = ds2.variables[var_name][:]
                    diff = np.abs(var - var2)
                    self.assertTrue(diff.max() <= 1.02734374963e-06)

    @attr('data')
    def test_to_netcdf_with_slice(self):
        rd = self.test_data.get_rd('narccap_rotated_pole')
        ops = OcgOperations(dataset=rd,
                            output_format='nc',
                            slice=[None, [0, 10], None, [0, 10], [0, 10]],
                            prefix='slice')
        ret = ops.execute()
        rd3 = ocgis.RequestDataset(uri=ret, variable='tas')
        self.assertEqual(rd3.get().shape, (1, 10, 1, 10, 10))


class Test(TestBase):
    @attr('data')
    def test_cf_lambert_conformal(self):
        rd = self.test_data.get_rd('narccap_lambert_conformal')
        field = rd.get()
        crs = field.spatial.crs
        self.assertDictEqual(crs.value, {'lon_0': -97, 'ellps': 'WGS84', 'y_0': 2700000, 'no_defs': True, 'proj': 'lcc',
                                         'x_0': 3325000, 'units': 'm', 'lat_2': 60, 'lat_1': 30, 'lat_0': 47.5})

    @attr('slow')
    def test_read_write_projections(self):
        """Test NARCCAP coordinate systems may be appropriately read and written to NetCDF."""

        data_dir = os.path.join(ocgis.env.DIR_TEST_DATA, 'nc', 'narccap')
        ocgis.env.DIR_DATA = data_dir
        ocgis.env.OVERWRITE = True

        real = {'pr': {'pr_RCM3_cgcm3_1986010103.nc': {'mu': 2.7800052478033606e-07, 'shape': (1, 1, 1, 7, 15)},
                       'pr_MM5I_ncep_1981010103.nc': {'mu': 3.3648159627007675e-08, 'shape': (1, 1, 1, 7, 14)},
                       'pr_RCM3_ncep_1986010103.nc': {'mu': 9.7176247154926553e-09, 'shape': (1, 1, 1, 7, 15)},
                       'pr_CRCM_ncep_1986010103.nc': {'mu': 1.1799650910219663e-26, 'shape': (1, 1, 1, 8, 16)},
                       'pr_CRCM_cgcm3_1981010103.nc': {'mu': 2.6299784818262446e-06, 'shape': (1, 1, 1, 8, 16)},
                       'pr_MM5I_ncep_1986010103.nc': {'mu': 0.0, 'shape': (1, 1, 1, 7, 14)},
                       'pr_HRM3_ncep_1981010103.nc': {'mu': 5.507401147596971e-10, 'shape': (1, 1, 1, 31, 22)},
                       'pr_RCM3_cgcm3_1981010103.nc': {'mu': 1.18896825283411e-05, 'shape': (1, 1, 1, 7, 15)},
                       'pr_TMSL_gfdl_1986010100.nc': {'mu': 2.0890602963450161e-07, 'shape': (1, 1, 1, 7, 15)},
                       'pr_WRFG_cgcm3_1986010103.nc': {'mu': 0.0, 'shape': (1, 1, 1, 7, 14)},
                       'pr_ECP2_gfdl_1981010103.nc': {'mu': 6.1180394635919263e-06, 'shape': (1, 1, 1, 9, 17)},
                       'pr_CRCM_ncep_1981010103.nc': {'mu': 2.767125774613198e-05, 'shape': (1, 1, 1, 8, 16)},
                       'pr_HRM3_gfdl_1986010103.nc': {'mu': 4.1377767579766305e-06, 'shape': (1, 1, 1, 31, 22)},
                       'pr_RCM3_gfdl_1981010103.nc': {'mu': -5.1954553518551086e-24, 'shape': (1, 1, 1, 7, 15)},
                       'pr_HRM3_ncep_1986010103.nc': {'mu': 0.0, 'shape': (1, 1, 1, 31, 22)},
                       'pr_TMSL_ccsm_1986010103.nc': {'mu': 3.734873736402074e-07, 'shape': (1, 1, 1, 7, 14)},
                       'pr_HRM3_gfdl_1981010103.nc': {'mu': 5.2488248024374339e-07, 'shape': (1, 1, 1, 31, 22)},
                       'pr_WRFG_ccsm_1986010103.nc': {'mu': 0.00010390303979970907, 'shape': (1, 1, 1, 7, 14)},
                       'pr_MM5I_ccsm_1986010103.nc': {'mu': 5.0342728890858494e-07, 'shape': (1, 1, 1, 7, 14)},
                       'pr_WRFG_ccsm_1981010103.nc': {'mu': np.ma.core.MaskedConstant, 'shape': (1, 1, 1, 7, 14)},
                       'pr_WRFG_cgcm3_1981010103.nc': {'mu': 0.0, 'shape': (1, 1, 1, 7, 14)},
                       'pr_WRFG_ncep_1981010103.nc': {'mu': np.ma.core.MaskedConstant, 'shape': (1, 1, 1, 7, 14)},
                       'pr_RCM3_ncep_1981010103.nc': {'mu': 7.637150009118376e-06, 'shape': (1, 1, 1, 7, 15)},
                       'pr_TMSL_ccsm_1981010103.nc': {'mu': 9.641077844117023e-27, 'shape': (1, 1, 1, 7, 14)},
                       'pr_RCM3_gfdl_1986010103.nc': {'mu': 1.0929620614097941e-05, 'shape': (1, 1, 1, 7, 15)},
                       'pr_TMSL_gfdl_1981010100.nc': {'mu': 1.3174895956811014e-10, 'shape': (1, 1, 1, 7, 15)},
                       'pr_CRCM_ccsm_1981010103.nc': {'mu': 1.6264247653238914e-06, 'shape': (1, 1, 1, 8, 16)},
                       'pr_WRFG_ncep_1986010103.nc': {'mu': np.ma.core.MaskedConstant, 'shape': (1, 1, 1, 7, 14)},
                       'pr_CRCM_cgcm3_1986010103.nc': {'mu': 3.152432755621917e-06, 'shape': (1, 1, 1, 8, 16)},
                       'pr_MM5I_ccsm_1981010103.nc': {'mu': 1.5723979779044096e-09, 'shape': (1, 1, 1, 7, 14)},
                       'pr_CRCM_ccsm_1986010103.nc': {'mu': 1.1736681164406678e-05, 'shape': (1, 1, 1, 8, 16)},
                       'pr_ECP2_gfdl_1986010103.nc': {'mu': 9.865492043614972e-06, 'shape': (1, 1, 1, 9, 17)}},
                'tas': {'tas_TMSL_gfdl_1986010100.nc': {'mu': 272.1787109375, 'shape': (1, 1, 1, 7, 15)},
                        'tas_RCM3_gfdl_1986010103.nc': {'mu': 257.74983723958331, 'shape': (1, 1, 1, 7, 15)},
                        'tas_HRM3_ncep_1986010103.nc': {'mu': 272.10732660060978, 'shape': (1, 1, 1, 31, 22)},
                        'tas_WRFG_ccsm_1981010103.nc': {'mu': 259.1943359375, 'shape': (1, 1, 1, 7, 14)},
                        'tas_TMSL_ccsm_1986010103.nc': {'mu': 271.766502490942, 'shape': (1, 1, 1, 7, 14)},
                        'tas_CRCM_cgcm3_1981010103.nc': {'mu': 256.05007276348039, 'shape': (1, 1, 1, 8, 16)},
                        'tas_RCM3_ncep_1981010103.nc': {'mu': 275.49927920386904, 'shape': (1, 1, 1, 7, 15)},
                        'tas_RCM3_gfdl_1981010103.nc': {'mu': 264.29543340773807, 'shape': (1, 1, 1, 7, 15)},
                        'tas_CRCM_ncep_1986010103.nc': {'mu': 268.38143382352939, 'shape': (1, 1, 1, 8, 16)},
                        'tas_CRCM_cgcm3_1986010103.nc': {'mu': 271.96783088235293, 'shape': (1, 1, 1, 8, 16)},
                        'tas_CRCM_ccsm_1986010103.nc': {'mu': 262.36866191789215, 'shape': (1, 1, 1, 8, 16)},
                        'tas_WRFG_cgcm3_1986010103.nc': {'mu': 274.17369962993422, 'shape': (1, 1, 1, 7, 14)},
                        'tas_MM5I_ccsm_1986010103.nc': {'mu': 260.47268194901318, 'shape': (1, 1, 1, 7, 14)},
                        'tas_TMSL_ccsm_1981010103.nc': {'mu': 275.4296875, 'shape': (1, 1, 1, 7, 14)},
                        'tas_WRFG_ccsm_1986010103.nc': {'mu': 260.7568359375, 'shape': (1, 1, 1, 7, 14)},
                        'tas_RCM3_ncep_1986010103.nc': {'mu': 268.04431733630952, 'shape': (1, 1, 1, 7, 15)},
                        'tas_CRCM_ncep_1981010103.nc': {'mu': 273.35757506127453, 'shape': (1, 1, 1, 8, 16)},
                        'tas_ECP2_gfdl_1981010103.nc': {'mu': 261.00524662990193, 'shape': (1, 1, 1, 9, 17)},
                        'tas_HRM3_ncep_1981010103.nc': {'mu': 274.20317263719511, 'shape': (1, 1, 1, 31, 22)},
                        'tas_RCM3_cgcm3_1981010103.nc': {'mu': 270.27541387648807, 'shape': (1, 1, 1, 7, 15)},
                        'tas_TMSL_gfdl_1981010100.nc': {'mu': 274.62939453125, 'shape': (1, 1, 1, 7, 15)},
                        'tas_HRM3_gfdl_1981010103.nc': {'mu': 257.89143483231709, 'shape': (1, 1, 1, 31, 22)},
                        'tas_WRFG_cgcm3_1981010103.nc': {'mu': 266.04286595394734, 'shape': (1, 1, 1, 7, 14)},
                        'tas_MM5I_ccsm_1981010103.nc': {'mu': 266.87697882401318, 'shape': (1, 1, 1, 7, 14)},
                        'tas_WRFG_ncep_1981010103.nc': {'mu': 275.5278834292763, 'shape': (1, 1, 1, 7, 14)},
                        'tas_MM5I_ncep_1981010103.nc': {'mu': 273.7758275082237, 'shape': (1, 1, 1, 7, 14)},
                        'tas_ECP2_gfdl_1986010103.nc': {'mu': 267.40525428921569, 'shape': (1, 1, 1, 9, 17)},
                        'tas_MM5I_ncep_1986010103.nc': {'mu': 267.03638980263156, 'shape': (1, 1, 1, 7, 14)},
                        'tas_RCM3_cgcm3_1986010103.nc': {'mu': 272.37116350446428, 'shape': (1, 1, 1, 7, 15)},
                        'tas_WRFG_ncep_1986010103.nc': {'mu': 268.89250102796052, 'shape': (1, 1, 1, 7, 14)},
                        'tas_HRM3_gfdl_1986010103.nc': {'mu': 270.4895674542683, 'shape': (1, 1, 1, 31, 22)},
                        'tas_CRCM_ccsm_1981010103.nc': {'mu': 264.73590686274508, 'shape': (1, 1, 1, 8, 16)}}}

        for uri in os.listdir(data_dir):
            if uri.endswith('nc'):
                variable = uri.split('_')[0]
                for output_format in ['numpy', 'nc']:
                    rd = RequestDataset(uri=uri, variable=variable)

                    try:
                        ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[16], snippet=True,
                                            output_format=output_format)
                    except DefinitionValidationError:
                        # rotated pole may not be written to netCDF
                        crs = rd._get_crs_()
                        if isinstance(crs, CFRotatedPole):
                            continue
                        else:
                            raise

                    try:
                        ret = ops.execute()

                        if output_format == 'numpy':
                            ref = ret[16].values()[0].variables.values()[0].value
                            mu = np.ma.mean(ref)

                            r_mu = real[variable][uri]['mu']
                            try:
                                self.assertAlmostEqual(r_mu, mu, places=4)
                            except AssertionError:
                                self.assertEqual(r_mu, np.ma.core.MaskedConstant)
                                self.assertEqual(real[variable][uri]['shape'], ref.shape)

                        if output_format == 'nc':
                            with nc_scope(ret) as ds:
                                try:
                                    grid_mapping = ds.variables[rd.variable].grid_mapping
                                    self.assertTrue(grid_mapping in ds.variables)
                                except AttributeError:
                                    # time slice files do not have a projection
                                    self.assertTrue('_TMSL_' in rd.uri)

                    except ExtentError:
                        if 'ECP2_ncep' in rd.uri:
                            pass
                        else:
                            raise
