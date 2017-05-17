import itertools
from csv import DictReader
from datetime import datetime as dt

import fiona
import numpy as np
from shapely.geometry.point import Point

import ocgis
from ocgis import RequestDataset
from ocgis import constants
from ocgis.base import get_variable_names
from ocgis.constants import DimensionMapKey
from ocgis.exc import ExtentError, RequestValidationError
from ocgis.ops.core import OcgOperations
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.variable.crs import Spherical, CFLambertConformal


class TestCnrmCerfacs(TestBase):
    @property
    def rd(self):
        return self.test_data.get_rd('rotated_pole_cnrm_cerfacs')

    @attr('data')
    def test_system_subset(self):
        """Test data may be subsetted and that coordinate transformations return the same value arrays."""

        ops = OcgOperations(dataset=self.rd, output_format=constants.OutputFormatName.OCGIS, snippet=True,
                            geom='world_countries', select_ugid=[69])
        ret = ops.execute()

        # Assert some of the geometry values are masked
        actual = ret.get_element().grid.get_mask()
        self.assertTrue(actual.any())

        # Perform the operations but change the output coordinate system. The value arrays should be equivalent
        # regardless of coordinate transformation.
        ops2 = OcgOperations(dataset=self.rd, output_format=constants.OutputFormatName.OCGIS, snippet=True,
                             geom='world_countries', select_ugid=[69], output_crs=Spherical())
        ret2 = ops2.execute()

        # Value arrays should be the same
        ret_value = ret.get_element(variable_name='pr').get_value()
        ret2_value = ret2.get_element(variable_name='pr').get_value()
        self.assertNumpyAll(ret_value, ret2_value)
        # Grid coordinates should not be the same.
        ret_grid_value = ret.get_element().grid.get_value_stacked()
        ret2_grid_value = ret2.get_element().grid.get_value_stacked()
        diff = np.abs(ret_grid_value - ret2_grid_value)
        select = diff > 1
        self.assertTrue(select.all())

    @attr('data', 'slow')
    def test_system_subset_shp(self):
        """Test conversion to shapefile."""

        for ii, output_crs in enumerate([None, Spherical()]):
            output_format = constants.OutputFormatName.SHAPEFILE
            ops = OcgOperations(dataset=self.rd, output_format=output_format, snippet=True,
                                geom='world_countries', select_ugid=[69], output_crs=output_crs, prefix=str(ii))
            ret = ops.execute()

            with fiona.open(ret) as source:
                records = list(source)

            self.assertTrue(len(records) > 2000)


class Test(TestBase):
    @attr('data')
    def test_ichec_rotated_pole(self):
        # This point is far outside the domain.
        ocgis.env.OVERWRITE = True
        rd = self.test_data.get_rd('rotated_pole_ichec')
        for geom in [[-100., 45.], [-100, 45, -99, 46]]:
            ops = ocgis.OcgOperations(dataset=rd, output_format='nc', calc=[{'func': 'mean', 'name': 'mean'}],
                                      calc_grouping=['month'], geom=geom)
            with self.assertRaises(ExtentError):
                ops.execute()

    @attr('data')
    def test_empty_subset_multi_geometry_wrapping(self):
        # Adjacent state boundaries were causing an error with wrapping where a reference to the source field was being
        # updated.
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[5, 6, 7])
        ret = ops.execute()
        self.assertEqual(set(ret.children.keys()), {5, 6, 7})

    @attr('data')
    def test_seasonal_calc(self):
        """Test some calculations using a seasonal grouping."""

        calc = [{'func': 'mean', 'name': 'my_mean'}, {'func': 'std', 'name': 'my_std'}]
        calc_grouping = [[3, 4, 5]]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, calc_sample_size=True,
                                  geom='state_boundaries', select_ugid=[23])
        ret = ops.execute()
        self.assertEqual(ret.get_element(variable_name='n_my_std').get_masked_value().mean(), 920.0)
        self.assertEqual(ret.get_element(variable_name='my_std').shape, (1, 3, 3))

        calc = [{'func': 'mean', 'name': 'my_mean'}, {'func': 'std', 'name': 'my_std'}]
        calc_grouping = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, calc_sample_size=True,
                                  geom='state_boundaries', select_ugid=[23])
        ret = ops.execute()
        self.assertEqual(ret.get_element(variable_name='my_std').shape, (4, 3, 3))
        temporal = ret.get_element().temporal
        numtime = temporal.value_numtime.data
        numtime_actual = np.array([56955.0, 56680.0, 56771.0, 56863.0])
        self.assertNumpyAll(numtime, numtime_actual)

        calc = [{'func': 'mean', 'name': 'my_mean'}, {'func': 'std', 'name': 'my_std'}]
        calc_grouping = [[12, 1], [2, 3]]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, calc_sample_size=True,
                                  geom='state_boundaries', select_ugid=[23])
        ret = ops.execute()
        self.assertEqual(ret.get_element(variable_name='my_std').shape, (2, 3, 3))
        temporal = ret.get_element().temporal
        bounds_numtime = temporal.bounds.value_numtime.data
        bounds_numtime_actual = np.array([[55115.0, 58765.0], [55146.0, 58490.0]])
        self.assertNumpyAll(bounds_numtime, bounds_numtime_actual)

    @attr('data', 'slow')
    def test_selecting_single_value(self):
        rd = self.test_data.get_rd('cancm4_tas')
        lat_index = 32
        lon_index = 97
        with nc_scope(rd.uri) as ds:
            lat_value = ds.variables['lat'][lat_index]
            lon_value = ds.variables['lon'][lon_index]
            data_values = ds.variables['tas'][:, lat_index, lon_index]

        ops = ocgis.OcgOperations(dataset=rd, geom=[lon_value, lat_value])
        ret = ops.execute()
        actual = ret.get_element(variable_name='tas').get_masked_value()
        values = np.squeeze(actual)
        self.assertNumpyAll(data_values, values.data)
        self.assertFalse(np.any(values.mask))

        geom = Point(lon_value, lat_value).buffer(0.001)
        ops = ocgis.OcgOperations(dataset=rd, geom=geom)
        ret = ops.execute()
        actual = ret.get_element(variable_name='tas').get_masked_value()
        values = np.squeeze(actual)
        self.assertNumpyAll(data_values, values.data)
        self.assertFalse(np.any(values.mask))

        geom = Point(lon_value - 360., lat_value).buffer(0.001)
        ops = ocgis.OcgOperations(dataset=rd, geom=geom)
        ret = ops.execute()
        actual = ret.get_element(variable_name='tas').get_masked_value()
        values = np.squeeze(actual)
        self.assertNumpyAll(data_values, values.data)
        self.assertFalse(np.any(values.mask))

        geom = Point(lon_value - 360., lat_value).buffer(0.001)
        ops = ocgis.OcgOperations(dataset=rd, geom=geom, aggregate=True, spatial_operation='clip')
        ret = ops.execute()
        actual = ret.get_element(variable_name='tas').get_masked_value()
        values = np.squeeze(actual)
        self.assertNumpyAll(data_values, values.data)
        self.assertFalse(np.any(values.mask))

        ops = ocgis.OcgOperations(dataset=rd, geom=[lon_value, lat_value],
                                  search_radius_mult=0.1, output_format='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            values = np.squeeze(ds.variables['tas'][:])
            self.assertNumpyAll(data_values, values)

    @attr('data')
    def test_time_region_subset(self):

        _month = [[6, 7], [12], None, [1, 3, 8]]
        _year = [[2011], None, [2012], [2011, 2013]]

        def run_test(month, year):
            rd = self.test_data.get_rd('cancm4_rhs', kwds={'time_region': {'month': month, 'year': year}})

            ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries',
                                      select_ugid=[25])
            ret = ops.execute()

            ret = ret.get_element().time.value_datetime

            years = [dt.year for dt in ret.flat]
            months = [dt.month for dt in ret.flat]

            if year is not None:
                self.assertEqual(set(years), set(year))
            if month is not None:
                self.assertEqual(set(months), set(month))

        for month, year in itertools.product(_month, _year):
            run_test(month, year)

    @attr('data')
    def test_time_range_time_region_subset(self):
        time_range = [dt(2013, 1, 1), dt(2015, 12, 31)]
        time_region = {'month': [6, 7, 8], 'year': [2013, 2014]}
        kwds = {'time_range': time_range, 'time_region': time_region}
        rd = self.test_data.get_rd('cancm4_rhs', kwds=kwds)
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element()
        years = set([obj.year for obj in ref.temporal.value_datetime])
        self.assertFalse(2015 in years)

    @attr('data')
    def test_time_range_time_region_do_not_overlap(self):
        time_range = [dt(2013, 1, 1), dt(2015, 12, 31)]
        time_region = {'month': [6, 7, 8], 'year': [2013, 2014, 2018]}
        kwds = {'time_range': time_range, 'time_region': time_region}
        with self.assertRaises(RequestValidationError):
            self.test_data.get_rd('cancm4_rhs', kwds=kwds)

    @attr('data')
    def test_clip_aggregate(self):
        # This geometry was hanging.
        rd = self.test_data.get_rd('cancm4_tas', kwds={'time_region': {'year': [2003]}})
        field = rd.get()
        ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[14, 16],
                            aggregate=False, spatial_operation='clip',
                            output_format=constants.OutputFormatName.CSV_SHAPEFILE)
        ops.execute()

    @attr('data')
    def test_narccap_point_subset_small(self):
        dmap = {DimensionMapKey.X: {DimensionMapKey.VARIABLE: 'xc'},
                DimensionMapKey.Y: {DimensionMapKey.VARIABLE: 'yc'},
                DimensionMapKey.TIME: {DimensionMapKey.VARIABLE: 'time'}}
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep', kwds={'dimension_map': dmap})

        field = rd.get()
        self.assertIsInstance(field.crs, CFLambertConformal)
        self.assertIsNotNone(field.time)

        geom = [-97.74278, 30.26694]

        calc = [{'func': 'mean', 'name': 'mean'},
                {'func': 'median', 'name': 'median'},
                {'func': 'max', 'name': 'max'},
                {'func': 'min', 'name': 'min'}]
        calc_grouping = ['month', 'year']
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                                  output_format=constants.OutputFormatName.OCGIS, geom=geom, abstraction='point',
                                  snippet=False, allow_empty=False, output_crs=Spherical(),
                                  search_radius_mult=2.0)
        ret = ops.execute()
        ref = ret.get_element()
        actual = set(get_variable_names(ref.data_variables))
        self.assertEqual(actual, {'mean', 'median', 'max', 'min'})

    @attr('data')
    def test_bad_time_dimension(self):
        """Test not formatting the time dimension."""

        for output_format in [constants.OutputFormatName.OCGIS, constants.OutputFormatName.CSV,
                              constants.OutputFormatName.CSV_SHAPEFILE, constants.OutputFormatName.SHAPEFILE,
                              constants.OutputFormatName.NETCDF]:
            dataset = self.test_data.get_rd('snippet_seasonalbias')
            ops = OcgOperations(dataset=dataset, output_format=output_format, format_time=False, prefix=output_format)
            ret = ops.execute()

            if output_format == constants.OutputFormatName.OCGIS:
                actual = ret.get_element()
                self.assertFalse(actual.temporal.format_time)
                self.assertNumpyAll(actual.temporal.value_numtime.data,
                                    np.array([-712208.5, -712117., -712025., -711933.5]))
                self.assertNumpyAll(actual.temporal.bounds.value_numtime.data,
                                    np.array([[-712254., -712163.], [-712163., -712071.], [-712071., -711979.],
                                              [-711979., -711888.]]))

            if output_format == constants.OutputFormatName.CSV:
                with open(ret) as f:
                    reader = DictReader(f)
                    for row in reader:
                        self.assertTrue(all([row[k] == '' for k in ['YEAR', 'MONTH', 'DAY']]))
                        self.assertTrue(float(row['TIME']) < -50000)

            if output_format == constants.OutputFormatName.NETCDF:
                self.assertNcEqual(ret, dataset.uri, check_types=False,
                                   ignore_attributes={'global': ['history'], 'bounds_time': ['calendar', 'units'],
                                                      'bias': ['_FillValue', 'grid_mapping', 'units'],
                                                      'latitude': ['standard_name', 'units'],
                                                      'longitude': ['standard_name', 'units']},
                                   ignore_variables=['latitude_longitude'])

    @attr('data')
    def test_mfdataset_to_nc(self):
        rd = self.test_data.get_rd('maurer_2010_pr')
        ops = OcgOperations(dataset=rd, output_format='nc', calc=[{'func': 'mean', 'name': 'my_mean'}],
                            calc_grouping=['year'], geom='state_boundaries', select_ugid=[23])
        ret = ops.execute()
        field = RequestDataset(ret, 'my_mean').get()
        self.assertNumpyAll(field.temporal.get_value(), np.array([18444., 18809.]))
