import datetime
import os
import shutil
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime as dt

import fiona
import netCDF4 as nc
import numpy as np
from shapely.geometry.geo import shape

import ocgis
from ocgis import GeomCabinet
from ocgis import RequestDataset
from ocgis import env
from ocgis.api.request.driver.nc import DriverNetcdf, get_dimension_map
from ocgis.exc import EmptySubsetError, DimensionNotFound, OcgWarning
from ocgis.interface.base.crs import WGS84, CFWGS84, CFLambertConformal
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGeometryPolygonDimension, SpatialGeometryDimension, \
    SpatialDimension
from ocgis.interface.metadata import NcMetadata
from ocgis.interface.nc.spatial import NcSpatialGridDimension
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.util.units import get_units_object


class TestDriverNetcdf(TestBase):
    def get_2d_state_boundaries(self):
        geoms = []
        build = True
        sc = GeomCabinet()
        path = sc.get_shp_path('state_boundaries')
        with fiona.open(path, 'r') as source:
            for ii, row in enumerate(source):
                if build:
                    nrows = len(source)
                    dtype = []
                    for k, v in source.schema['properties'].iteritems():
                        if v.startswith('str'):
                            v = str('|S{0}'.format(v.split(':')[1]))
                        else:
                            v = getattr(np, v.split(':')[0])
                        dtype.append((str(k), v))
                    fill = np.empty(nrows, dtype=dtype)
                    ref_names = fill.dtype.names
                    build = False
                fill[ii] = tuple([row['properties'][n] for n in ref_names])
                geoms.append(shape(row['geometry']))
        geoms = np.atleast_2d(geoms)
        return geoms, fill

    def get_2d_state_boundaries_sdim(self):
        geoms, attrs = self.get_2d_state_boundaries()
        poly = SpatialGeometryPolygonDimension(value=geoms)
        geom = SpatialGeometryDimension(polygon=poly)
        sdim = SpatialDimension(geom=geom, properties=attrs, crs=WGS84())
        return sdim

    @attr('data')
    def test_get_dimensioned_variables_one_variable_in_target_dataset(self):
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=uri)
        driver = DriverNetcdf(rd)
        ret = driver.get_dimensioned_variables()
        self.assertEqual(ret, ['tas'])
        self.assertEqual(rd.variable, ret[0])

    @attr('data')
    def test_get_dimensioned_variables_two_variables_in_target_dataset(self):
        rd_orig = self.test_data.get_rd('cancm4_tas')
        dest_uri = os.path.join(self.current_dir_output, os.path.split(rd_orig.uri)[1])
        shutil.copy2(rd_orig.uri, dest_uri)
        with nc_scope(dest_uri, 'a') as ds:
            var = ds.variables['tas']
            outvar = ds.createVariable(var._name + 'max', var.dtype, var.dimensions)
            outvar[:] = var[:] + 3
            outvar.setncatts(var.__dict__)
        rd = RequestDataset(uri=dest_uri)
        self.assertEqual(rd.variable, ('tas', 'tasmax'))
        self.assertEqual(rd.variable, rd.alias)

    @attr('data')
    def test_get_dump_report(self):
        rd = self.test_data.get_rd('cancm4_tas')
        driver = DriverNetcdf(rd)
        self.assertTrue(len(driver.get_dump_report()) > 15)

    @attr('data')
    def test_get_field(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri)
        field = rd.get()

        self.assertIsInstance(field.spatial.grid, NcSpatialGridDimension)

        # Test names are correctly set when creating the field.
        self.assertEqual(field.temporal.name, 'time')
        self.assertEqual(field.temporal.name_value, 'time')
        self.assertEqual(field.temporal.name_bounds, 'time_bnds')
        row = field.spatial.grid.row
        self.assertEqual(row.name, 'lat')
        self.assertEqual(row.name_value, 'lat')
        self.assertEqual(row.name_bounds, 'lat_bnds')
        col = field.spatial.grid.col
        self.assertEqual(col.name, 'lon')
        self.assertEqual(col.name_value, 'lon')
        self.assertEqual(col.name_bounds, 'lon_bnds')

        # Test attributes are loaded.
        self.assertEqual(len(field.attrs), 31)
        self.assertEqual(len(field.variables['tas'].attrs), 10)
        self.assertEqual(len(field.temporal.attrs), 6)
        self.assertEqual(len(field.spatial.grid.row.attrs), 5)
        self.assertEqual(len(field.spatial.grid.col.attrs), 5)

        with self.nc_scope(uri) as ds:
            self.assertEqual(field.level, None)
            self.assertEqual(field.spatial.crs, WGS84())

            tv = field.temporal.value
            test_tv = ds.variables['time'][:]
            self.assertNumpyAll(tv, test_tv)
            self.assertNumpyAll(field.temporal.bounds, ds.variables['time_bnds'][:])

            tdt = field.temporal.value_datetime
            self.assertEqual(tdt[4], dt(2001, 1, 5, 12))
            self.assertNumpyAll(field.temporal.bounds_datetime[1001], np.array([dt(2003, 9, 29), dt(2003, 9, 30)]))

            rv = field.temporal.value_datetime[100]
            rb = field.temporal.bounds_datetime[100]
            self.assertTrue(all([rv > rb[0], rv < rb[1]]))

            self.assertEqual(field.temporal.extent_datetime,
                             (datetime.datetime(2001, 1, 1), datetime.datetime(2011, 1, 1)))

    @attr('data')
    def test_get_field_t_conform_units_to(self):
        """
        Test conforming time units is appropriately passed to field object.
        """

        uri = self.test_data.get_uri('cancm4_tas')
        target = get_units_object('days since 1949-1-1', calendar='365_day')
        rd = RequestDataset(uri=uri, t_conform_units_to=target)
        field = rd.get()
        self.assertEqual(field.temporal.conform_units_to, target)

    def test_get_field_different_dimension_names_and_values(self):
        """Test dimension names and dimension values are correctly read from netCDF."""

        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            ds.createDimension('lat', 1)
            ds.createDimension('lon', 1)
            ds.createDimension('tme', 1)
            ds.createDimension('the_bounds', 2)
            latitude = ds.createVariable('latitude', int, dimensions=('lat',))
            longitude = ds.createVariable('longitude', int, dimensions=('lon',))
            time = ds.createVariable('time', int, dimensions=('tme',))
            time_bounds = ds.createVariable('long_live_the_bounds', int, dimensions=('tme', 'the_bounds'))
            time.units = 'days since 0000-01-01'
            time.bounds = 'long_live_the_bounds'
            value = ds.createVariable('value', int, dimensions=('tme', 'lat', 'lon'))

            latitude[:] = 5
            longitude[:] = 6
            time[:] = 6
            value[:] = np.array([7]).reshape(1, 1, 1)

        rd = RequestDataset(path)
        driver = DriverNetcdf(rd)
        field = driver._get_field_()
        self.assertEqual(field.temporal.name, 'tme')
        self.assertEqual(field.temporal.name_value, 'time')
        self.assertEqual(field.spatial.grid.row.name, 'lat')
        self.assertEqual(field.spatial.grid.row.name_value, 'latitude')
        self.assertEqual(field.spatial.grid.col.name, 'lon')
        self.assertEqual(field.spatial.grid.col.name_value, 'longitude')
        self.assertEqual(field.temporal.name_bounds, 'long_live_the_bounds')

    @attr('data')
    def test_get_field_dtype_on_dimensions(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        with nc_scope(rd.uri) as ds:
            test_dtype_temporal = ds.variables['time'].dtype
            test_dtype_value = ds.variables['tas'].dtype
        self.assertEqual(field.temporal.dtype, test_dtype_temporal)
        self.assertEqual(field.variables['tas'].dtype, test_dtype_value)
        self.assertEqual(field.temporal.dtype, np.float64)

    @attr('data')
    def test_get_field_dtype_fill_value(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        # dtype and fill_value should be read from metadata. when accessed they should not load the value.
        self.assertEqual(field.variables['tas'].dtype, np.float32)
        self.assertEqual(field.variables['tas'].fill_value, np.float32(1e20))
        self.assertEqual(field.variables['tas']._value, None)

    @attr('data')
    def test_get_field_datetime_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri)
        field = rd.get()

        field.temporal.value_datetime
        field.temporal.bounds_datetime

        slced = field[:, 239, :, :, :]
        self.assertEqual(slced.temporal.value_datetime, np.array([dt(2001, 8, 28, 12)]))
        self.assertNumpyAll(slced.temporal.bounds_datetime, np.array([dt(2001, 8, 28), dt(2001, 8, 29)]).reshape(1, 2))

    @attr('data')
    def test_get_field_units_read_from_file(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertEqual(field.variables['tas'].cfunits, get_units_object('K'))

    @attr('data')
    def test_get_field_value_datetime_after_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri)
        field = rd.get()
        slced = field[:, 10:130, :, 4:7, 100:37]
        self.assertEqual(slced.temporal.value_datetime.shape, (120,))

    @attr('data')
    def test_get_field_bounds_datetime_after_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri)
        field = rd.get()
        slced = field[:, 10:130, :, 4:7, 100:37]
        self.assertEqual(slced.temporal.bounds_datetime.shape, (120, 2))

    @attr('data')
    def test_get_field_slice(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri)
        field = rd.get()
        ds = nc.Dataset(uri, 'r')

        slced = field[:, 56:345, :, :, :]
        self.assertNumpyAll(slced.temporal.value, ds.variables['time'][56:345])
        self.assertNumpyAll(slced.temporal.bounds, ds.variables['time_bnds'][56:345, :])
        to_test = ds.variables['tas'][56:345, :, :]
        to_test = np.ma.array(to_test.reshape(1, 289, 1, 64, 128), mask=False)
        self.assertNumpyAll(slced.variables['tas'].value, to_test)

        slced = field[:, 2898, :, 5, 101]
        to_test = ds.variables['tas'][2898, 5, 101]
        to_test = np.ma.array(to_test.reshape(1, 1, 1, 1, 1), mask=False)
        with self.assertRaises(AttributeError):
            slced.variables['tas']._field._value
        self.assertNumpyAll(slced.variables['tas'].value, to_test)

        ds.close()

    @attr('data')
    def test_get_field_time_range(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri, time_range=[dt(2005, 2, 15), dt(2007, 4, 18)])
        field = rd.get()
        self.assertEqual(field.temporal.value_datetime[0], dt(2005, 2, 15, 12, 0))
        self.assertEqual(field.temporal.value_datetime[-1], dt(2007, 4, 18, 12, 0))
        self.assertEqual(field.shape, (1, 793, 1, 64, 128))

    @attr('data')
    def test_get_field_time_region(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        ds = nc.Dataset(uri, 'r')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri, time_region={'month': [8]})
        field = rd.get()

        self.assertEqual(field.shape, (1, 310, 1, 64, 128))

        var = ds.variables['time']
        real_temporal = nc.num2date(var[:], var.units, var.calendar)
        select = [True if x.month == 8 else False for x in real_temporal]
        indices = np.arange(0, var.shape[0], dtype=np.int32)[np.array(select)]
        self.assertNumpyAll(indices, field.temporal._src_idx)
        self.assertNumpyAll(field.temporal.value_datetime, real_temporal[indices])
        self.assertNumpyAll(field.variables['tas'].value.data.squeeze(), ds.variables['tas'][indices, :, :])

        bounds_temporal = nc.num2date(ds.variables['time_bnds'][indices, :], var.units, var.calendar)
        self.assertNumpyAll(bounds_temporal, field.temporal.bounds_datetime)

        ds.close()

    @attr('data')
    def test_get_field_time_region_with_years(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        ds = nc.Dataset(uri, 'r')
        rd = RequestDataset(variable=ref_test['variable'], uri=uri, time_region={'month': [8], 'year': [2008, 2010]})
        field = rd.get()

        self.assertEqual(field.shape, (1, 62, 1, 64, 128))

        var = ds.variables['time']
        real_temporal = nc.num2date(var[:], var.units, var.calendar)
        select = [True if x.month == 8 and x.year in [2008, 2010] else False for x in real_temporal]
        indices = np.arange(0, var.shape[0], dtype=np.int32)[np.array(select)]
        self.assertNumpyAll(indices, field.temporal._src_idx)
        self.assertNumpyAll(field.temporal.value_datetime, real_temporal[indices])
        self.assertNumpyAll(field.variables['tas'].value.data.squeeze(), ds.variables['tas'][indices, :, :])

        bounds_temporal = nc.num2date(ds.variables['time_bnds'][indices, :], var.units, var.calendar)
        self.assertNumpyAll(bounds_temporal, field.temporal.bounds_datetime)

        ds.close()

    @attr('data')
    def test_get_field_geometry_subset(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')

        states = self.get_2d_state_boundaries_sdim()
        ca = states[:, states.properties['STATE_NAME'] == 'California']
        self.assertTrue(ca.properties['STATE_NAME'] == 'California')
        ca.crs.unwrap(ca)
        ca = ca.geom.polygon.value[0, 0]

        for u in [True, False]:
            rd = RequestDataset(variable=ref_test['variable'], uri=uri, alias='foo')
            field = rd.get()
            ca_sub = field.get_intersects(ca, use_spatial_index=u)
            self.assertEqual(ca_sub.shape, (1, 3650, 1, 5, 4))
            self.assertTrue(ca_sub.variables['foo'].value.mask.any())
            self.assertFalse(field.spatial.uid.mask.any())
            self.assertFalse(field.spatial.get_mask().any())

            ca_sub = field.get_intersects(ca.envelope, use_spatial_index=u)
            self.assertEqual(ca_sub.shape, (1, 3650, 1, 5, 4))
            self.assertFalse(ca_sub.variables['foo'].value.mask.any())

            rd = RequestDataset(variable=ref_test['variable'], uri=uri, alias='foo', time_region={'year': [2007]})
            field = rd.get()
            ca_sub = field.get_intersects(ca, use_spatial_index=u)
            self.assertEqual(ca_sub.shape, (1, 365, 1, 5, 4))
            self.assertEqual(set([2007]), set([d.year for d in ca_sub.temporal.value_datetime]))

    @attr('data')
    def test_get_field_time_region_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')

        rd = RequestDataset(variable=ref_test['variable'], uri=uri, alias='foo',
                            time_region={'month': [1, 10], 'year': [2011, 2013]})
        with self.assertRaises(EmptySubsetError):
            rd.get()

        rd = RequestDataset(variable=ref_test['variable'], uri=uri, alias='foo',
                            time_region={'month': [1, 10], 'year': [2005, 2007]})
        field = rd.get()
        sub = field[:, :, :, 50, 75]
        self.assertEqual(sub.shape, (1, 124, 1, 1, 1))
        self.assertEqual(sub.variables['foo'].value.shape, (1, 124, 1, 1, 1))

        field = rd.get()
        sub = field[:, :, :, 50, 75:77]
        sub2 = field[:, :, :, 0, 1]
        self.assertEqual(sub2.shape, (1, 124, 1, 1, 1))

    @attr('remote', 'slow')
    def test_get_field_remote(self):
        uri = 'http://cida.usgs.gov/thredds/dodsC/maurer/maurer_brekke_w_meta.ncml'
        variable = 'sresa1b_bccr-bcm2-0_1_Tavg'
        rd = RequestDataset(uri, variable, time_region={'month': [1, 10], 'year': [2011, 2013]})
        field = rd.get()
        field.variables['sresa1b_bccr-bcm2-0_1_Tavg'].value
        values = field[:, :, :, 50, 75]
        to_test = values.variables['sresa1b_bccr-bcm2-0_1_Tavg'].value.compressed()

        ds = nc.Dataset('http://cida.usgs.gov/thredds/dodsC/maurer/maurer_brekke_w_meta.ncml', 'r')
        try:
            values = ds.variables['sresa1b_bccr-bcm2-0_1_Tavg'][:, 50, 75]
            times = nc.num2date(ds.variables['time'][:], ds.variables['time'].units, ds.variables['time'].calendar)
            select = np.array([True if time in list(field.temporal.value_datetime) else False for time in times])
            sel_values = values[select]
            self.assertNumpyAll(to_test, sel_values)
        finally:
            ds.close()

    @attr('data')
    def test_get_field_with_projection(self):
        uri = self.test_data.get_uri('narccap_wrfg')
        rd = RequestDataset(uri, 'pr')
        field = rd.get()
        self.assertIsInstance(field.spatial.crs, CFLambertConformal)
        field.spatial.update_crs(CFWGS84())
        self.assertIsInstance(field.spatial.crs, CFWGS84)
        self.assertEqual(field.spatial.grid.row, None)
        self.assertAlmostEqual(field.spatial.grid.value.mean(), -26.269666952512416)
        field.spatial.crs.unwrap(field.spatial)
        self.assertAlmostEqual(field.spatial.grid.value.mean(), 153.73033304748759)
        self.assertIsNone(field.spatial.geom.polygon)
        self.assertAlmostEqual(field.spatial.geom.point.value[0, 100].x, 278.52630062012787)
        self.assertAlmostEqual(field.spatial.geom.point.value[0, 100].y, 21.4615681252577)

    @attr('data')
    def test_get_field_projection_axes(self):
        uri = self.test_data.get_uri('cmip3_extraction')
        variable = 'Tavg'
        rd = RequestDataset(uri, variable)
        with self.assertRaises(DimensionNotFound):
            rd.get()
        rd = RequestDataset(uri, variable, dimension_map={'R': 'projection', 'T': 'time', 'X': 'longitude',
                                                          'Y': 'latitude'})
        field = rd.get()
        self.assertEqual(field.shape, (36, 1800, 1, 7, 12))
        self.assertEqual(field.temporal.value_datetime[0], datetime.datetime(1950, 1, 16, 0, 0))
        self.assertEqual(field.temporal.value_datetime[-1], datetime.datetime(2099, 12, 15, 0, 0))
        self.assertEqual(field.level, None)
        self.assertNumpyAll(field.realization.value, np.arange(1, 37, dtype=np.int32))

        ds = nc.Dataset(uri, 'r')
        to_test = ds.variables['Tavg']
        self.assertNumpyAll(to_test[:], field.variables['Tavg'].value.squeeze())
        ds.close()

    @attr('data')
    def test_get_field_projection_axes_slicing(self):
        uri = self.test_data.get_uri('cmip3_extraction')
        variable = 'Tavg'
        rd = RequestDataset(uri, variable,
                            dimension_map={'R': 'projection', 'T': 'time', 'X': 'longitude', 'Y': 'latitude'})
        field = rd.get()
        sub = field[15, :, :, :, :]
        self.assertEqual(sub.shape, (1, 1800, 1, 7, 12))

        ds = nc.Dataset(uri, 'r')
        to_test = ds.variables['Tavg']
        self.assertNumpyAll(to_test[15, :, :, :], sub.variables[variable].value.squeeze())
        ds.close()

    @attr('data')
    def test_get_field_multifile_load(self):
        uri = self.test_data.get_uri('narccap_pr_wrfg_ncep')
        rd = RequestDataset(uri, 'pr')
        field = rd.get()
        self.assertEqual(field.temporal.extent_datetime,
                         (datetime.datetime(1981, 1, 1, 0, 0), datetime.datetime(1991, 1, 1, 0, 0)))
        self.assertAlmostEqual(field.temporal.resolution, 0.125)

    @attr('data')
    def test_get_field_climatology_bounds(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, output_format='nc', geom='state_boundaries',
                                  select_ugid=[27], calc=[{'func': 'mean', 'name': 'mean'}],
                                  calc_grouping=['month'])
        ret = ops.execute()
        rd = RequestDataset(uri=ret, variable='mean')
        field = rd.get()
        self.assertIsNotNone(field.temporal.bounds)

    def test_get_field_without_row_column_vectors(self):
        """Test loading a field object without row and column vectors."""

        path = self.get_netcdf_path_no_row_column()

        rd = RequestDataset(path)
        driver = DriverNetcdf(rd)
        new_field = driver.get_field()
        self.assertIsNotNone(new_field.crs)
        grid = new_field.spatial.grid
        self.assertIsNone(grid.row)
        self.assertIsNone(grid.col)
        self.assertEqual(grid.name_row, 'yc')
        self.assertEqual(grid.name_col, 'xc')
        self.assertIsNone(grid._value)
        actual = np.ma.array([[[4.0, 4.0], [5.0, 5.0]], [[40.0, 50.0], [40.0, 50.0]]])
        self.assertNumpyAll(grid.value, actual)
        var = new_field.variables.first()
        self.assertEqual(var.shape, (1, 2, 1, 2, 2))
        self.assertEqual(var.value.shape, (1, 2, 1, 2, 2))

        new_field = driver.get_field()
        grid = new_field.spatial.grid
        self.assertEqual(grid.shape, (2, 2))
        self.assertIsNone(grid._value)
        sub = new_field[:, :, :, 0, 1]
        sub_grid = sub.spatial.grid
        self.assertIsInstance(sub_grid, NcSpatialGridDimension)
        self.assertEqual(sub.shape, (1, 2, 1, 1, 1))
        self.assertIsNone(sub_grid._value)
        self.assertEqual(sub_grid.shape, (1, 1))
        actual = np.ma.array([[[4.0]], [[50.0]]])
        self.assertNumpyAll(actual, sub_grid.value)
        sub_var = sub.variables.first()
        self.assertEqual(sub_var.shape, (1, 2, 1, 1, 1))
        self.assertEqual(sub_var.value.shape, (1, 2, 1, 1, 1))

        path2 = os.path.join(self.current_dir_output, 'foo2.nc')
        with self.nc_scope(path2, 'w') as ds:
            new_field.write_netcdf(ds)
        self.assertNcEqual(path, path2, ignore_attributes={'foo': ['grid_mapping']},
                           ignore_variables=['latitude_longitude'])

    @attr('data')
    def test_get_name_bounds_dimension(self):
        rd = self.test_data.get_rd('cancm4_tas')
        source_metadata = rd.source_metadata
        res = DriverNetcdf._get_name_bounds_dimension_(source_metadata)
        self.assertEqual(res, 'bnds')

        # remove any mention of bounds from the dimension map and try again
        for value in source_metadata['dim_map'].itervalues():
            try:
                value['bounds'] = None
            except TypeError:
                # likely a nonetype
                if value is not None:
                    raise
        res = DriverNetcdf._get_name_bounds_dimension_(source_metadata)
        self.assertIsNone(res)

        # now remove the bounds key completely
        for value in source_metadata['dim_map'].itervalues():
            try:
                value.pop('bounds')
            except AttributeError:
                if value is not None:
                    raise
        res = DriverNetcdf._get_name_bounds_dimension_(source_metadata)
        self.assertIsNone(res)

    @attr('data')
    def test_get_source_metadata(self):
        dimension_map = {'X': {'variable': 'lon', 'dimension': 'x', 'pos': 2, 'bounds': 'lon_bnds'},
                         'Y': {'variable': 'lat', 'dimension': 'y', 'pos': 1, 'bounds': 'lat_bounds'},
                         'T': {'variable': 'time', 'dimension': 'time', 'pos': 0, 'bounds': 'time_bounds'}}
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=uri, dimension_map=dimension_map)
        driver = DriverNetcdf(rd)
        meta = driver.get_source_metadata()
        self.assertEqual(meta['dim_map'], dimension_map)

        # test with no dimensioned variables
        uri = self.get_netcdf_path_no_dimensioned_variables()
        rd = RequestDataset(uri=uri)
        driver = DriverNetcdf(rd)
        meta = driver.get_source_metadata()
        self.assertEqual(meta['dimensions']['dim']['len'], 0)
        with self.assertRaises(KeyError):
            meta['dim_map']

    @attr('data')
    def test_source_metadata_as_json(self):
        rd = self.test_data.get_rd('cancm4_tas')
        driver = DriverNetcdf(rd)
        js = driver.get_source_metadata_as_json()
        self.assertIsInstance(js, basestring)
        self.assertTrue(len(js) > 100)

    def test_get_vector_dimension(self):
        # test exception raised with no row and column
        path = self.get_netcdf_path_no_row_column()
        rd = RequestDataset(path)
        driver = DriverNetcdf(rd)
        k = 'row'
        v = {'name_uid': 'yc_id', 'axis': 'Y', 'adds': {'interpolate_bounds': False}, 'name': 'yc',
             'cls': VectorDimension}
        source_metadata = rd.source_metadata
        res = driver._get_vector_dimension_(k, v, source_metadata)
        self.assertEqual(res['name'], 'yc')

    @attr('data')
    def test_inspect(self):
        rd = self.test_data.get_rd('cancm4_tas')
        driver = DriverNetcdf(rd)
        with self.print_scope() as ps:
            driver.inspect()
        self.assertTrue(len(ps.storage) >= 1)

        # test with a request dataset having no dimensioned variables
        path = self.get_temporary_file_path('bad.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('foo')
            var = ds.createVariable('foovar', int, dimensions=('foo',))
            var.a_name = 'a name'
        rd = RequestDataset(uri=path)
        driver = DriverNetcdf(rd)
        with self.print_scope() as ps:
            driver.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    @attr('data')
    def test_open(self):
        # test a multifile dataset where the variable does not appear in all datasets
        uri1 = self.test_data.get_uri('cancm4_tas')
        uri2 = self.test_data.get_uri('cancm4_tasmax_2001')
        uri = [uri1, uri2]
        rd = RequestDataset(uri=uri, variable='tas')
        driver = DriverNetcdf(rd)
        with self.assertRaises(KeyError):
            driver.open()
        with self.assertRaises(KeyError):
            rd.source_metadata


class Test(TestBase):
    @attr('data')
    def test_get_dimension_map_1(self):
        """Test dimension dictionary returned correctly."""

        rd = self.test_data.get_rd('cancm4_tas')
        dim_map = get_dimension_map('tas', rd.source_metadata)
        self.assertDictEqual(dim_map, {'Y': {'variable': u'lat', 'bounds': u'lat_bnds', 'dimension': u'lat', 'pos': 1},
                                       'X': {'variable': u'lon', 'bounds': u'lon_bnds', 'dimension': u'lon', 'pos': 2},
                                       'Z': None,
                                       'T': {'variable': u'time', 'bounds': u'time_bnds', 'dimension': u'time',
                                             'pos': 0}})

    def test_get_dimension_map_2(self):
        """Test special case where bounds were in the file but not found by the code."""

        # this metadata was causing an issue with the bounds not being discovered (Maurer02new_OBS_tas_daily.1971-2000.nc)
        # rd = RequestDataset(uri='/usr/local/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tas_daily.1971-2000.nc', variable='tas')
        # metadata = rd.source_metadata
        metadata = NcMetadata([('dataset', OrderedDict(
            [(u'CDI', u'Climate Data Interface version 1.5.0 (http://code.zmaw.de/projects/cdi)'),
             (u'Conventions', u'GDT 1.2'), (u'history',
                                            u'Wed Jul  3 07:17:09 2013: ncrcat nldas_met_update.obs.daily.tas.1971.nc nldas_met_update.obs.daily.tas.1972.nc nldas_met_update.obs.daily.tas.1973.nc nldas_met_update.obs.daily.tas.1974.nc nldas_met_update.obs.daily.tas.1975.nc nldas_met_update.obs.daily.tas.1976.nc nldas_met_update.obs.daily.tas.1977.nc nldas_met_update.obs.daily.tas.1978.nc nldas_met_update.obs.daily.tas.1979.nc nldas_met_update.obs.daily.tas.1980.nc nldas_met_update.obs.daily.tas.1981.nc nldas_met_update.obs.daily.tas.1982.nc nldas_met_update.obs.daily.tas.1983.nc nldas_met_update.obs.daily.tas.1984.nc nldas_met_update.obs.daily.tas.1985.nc nldas_met_update.obs.daily.tas.1986.nc nldas_met_update.obs.daily.tas.1987.nc nldas_met_update.obs.daily.tas.1988.nc nldas_met_update.obs.daily.tas.1989.nc nldas_met_update.obs.daily.tas.1990.nc nldas_met_update.obs.daily.tas.1991.nc nldas_met_update.obs.daily.tas.1992.nc nldas_met_update.obs.daily.tas.1993.nc nldas_met_update.obs.daily.tas.1994.nc nldas_met_update.obs.daily.tas.1995.nc nldas_met_update.obs.daily.tas.1996.nc nldas_met_update.obs.daily.tas.1997.nc nldas_met_update.obs.daily.tas.1998.nc nldas_met_update.obs.daily.tas.1999.nc nldas_met_update.obs.daily.tas.2000.nc Maurer02new_OBS_tas_daily.1971-2000.nc\nFri Oct 28 08:44:48 2011: cdo ifthen conus_mask.nc /archive/public/gridded_obs/daily/ncfiles_2010/nldas_met_update.obs.daily.tas.1971.nc /data3/emaurer/ldas_met_2010/process/met/nc/daily/nldas_met_update.obs.daily.tas.1971.nc'),
             (u'institution', u'Princeton U.'), (u'file_name', u'nldas_met_update.obs.daily.tas.1971.nc'),
             (u'History', u'Interpolated from 1-degree data'), (u'authors',
                                                                u'Sheffield, J., G. Goteti, and E. F. Wood, 2006: Development of a 50-yr high-resolution global dataset of meteorological forcings for land surface modeling, J. Climate, 19 (13), 3088-3111'),
             (u'description', u'Gridded Observed global data'), (u'creation_date', u'2006'),
             (u'SurfSgnConvention', u'Traditional'),
             (u'CDO', u'Climate Data Operators version 1.5.0 (http://code.zmaw.de/projects/cdo)'),
             (u'nco_openmp_thread_number', 1)])), ('file_format', 'NETCDF3_CLASSIC'), ('variables', OrderedDict([(
                                                                                                                     u'longitude',
                                                                                                                     {
                                                                                                                         'dtype': 'float32',
                                                                                                                         'fill_value': 1e+20,
                                                                                                                         'dimensions': (
                                                                                                                             u'longitude',),
                                                                                                                         'name': u'longitude',
                                                                                                                         'attrs': OrderedDict(
                                                                                                                             [
                                                                                                                                 (
                                                                                                                                     u'long_name',
                                                                                                                                     u'Longitude'),
                                                                                                                                 (
                                                                                                                                     u'units',
                                                                                                                                     u'degrees_east'),
                                                                                                                                 (
                                                                                                                                     u'standard_name',
                                                                                                                                     u'longitude'),
                                                                                                                                 (
                                                                                                                                     u'axis',
                                                                                                                                     u'X'),
                                                                                                                                 (
                                                                                                                                     u'bounds',
                                                                                                                                     u'longitude_bnds')])}),
                                                                                                                 (
                                                                                                                     u'longitude_bnds',
                                                                                                                     {
                                                                                                                         'dtype': 'float32',
                                                                                                                         'fill_value': 1e+20,
                                                                                                                         'dimensions': (
                                                                                                                             u'longitude',
                                                                                                                             u'nb2'),
                                                                                                                         'name': u'longitude_bnds',
                                                                                                                         'attrs': OrderedDict()}),
                                                                                                                 (
                                                                                                                     u'latitude',
                                                                                                                     {
                                                                                                                         'dtype': 'float32',
                                                                                                                         'fill_value': 1e+20,
                                                                                                                         'dimensions': (
                                                                                                                             u'latitude',),
                                                                                                                         'name': u'latitude',
                                                                                                                         'attrs': OrderedDict(
                                                                                                                             [
                                                                                                                                 (
                                                                                                                                     u'long_name',
                                                                                                                                     u'Latitude'),
                                                                                                                                 (
                                                                                                                                     u'units',
                                                                                                                                     u'degrees_north'),
                                                                                                                                 (
                                                                                                                                     u'standard_name',
                                                                                                                                     u'latitude'),
                                                                                                                                 (
                                                                                                                                     u'axis',
                                                                                                                                     u'Y'),
                                                                                                                                 (
                                                                                                                                     u'bounds',
                                                                                                                                     u'latitude_bnds')])}),
                                                                                                                 (
                                                                                                                     u'latitude_bnds',
                                                                                                                     {
                                                                                                                         'dtype': 'float32',
                                                                                                                         'fill_value': 1e+20,
                                                                                                                         'dimensions': (
                                                                                                                             u'latitude',
                                                                                                                             u'nb2'),
                                                                                                                         'name': u'latitude_bnds',
                                                                                                                         'attrs': OrderedDict()}),
                                                                                                                 (
                                                                                                                     u'time',
                                                                                                                     {
                                                                                                                         'dtype': 'float64',
                                                                                                                         'fill_value': 1e+20,
                                                                                                                         'dimensions': (
                                                                                                                             u'time',),
                                                                                                                         'name': u'time',
                                                                                                                         'attrs': OrderedDict(
                                                                                                                             [
                                                                                                                                 (
                                                                                                                                     u'units',
                                                                                                                                     u'days since 1940-01-01 00:00:00'),
                                                                                                                                 (
                                                                                                                                     u'calendar',
                                                                                                                                     u'standard')])}),
                                                                                                                 (
                                                                                                                     u'tas',
                                                                                                                     {
                                                                                                                         'dtype': 'float32',
                                                                                                                         'fill_value': 1e+20,
                                                                                                                         'dimensions': (
                                                                                                                             u'time',
                                                                                                                             u'latitude',
                                                                                                                             u'longitude'),
                                                                                                                         'name': u'tas',
                                                                                                                         'attrs': OrderedDict(
                                                                                                                             [
                                                                                                                                 (
                                                                                                                                     u'units',
                                                                                                                                     u'C')])})])),
                               ('dimensions', OrderedDict([(u'longitude', {'isunlimited': False, 'len': 462}),
                                                           (u'nb2', {'isunlimited': False, 'len': 2}),
                                                           (u'latitude', {'isunlimited': False, 'len': 222}),
                                                           (u'time', {'isunlimited': True, 'len': 10958})]))])
        dim_map = get_dimension_map('tas', metadata)
        self.assertDictEqual(dim_map, {
            'Y': {'variable': u'latitude', 'bounds': u'latitude_bnds', 'dimension': u'latitude', 'pos': 1},
            'X': {'variable': u'longitude', 'bounds': u'longitude_bnds', 'dimension': u'longitude', 'pos': 2},
            'Z': None,
            'T': {'variable': u'time', 'bounds': None, 'dimension': u'time', 'pos': 0}})

    @attr('data')
    def test_get_dimension_map_3(self):
        """Test when bounds are found but the bounds variable is actually missing."""

        # remove the bounds variable from a standard metadata dictionary
        def _run_():
            env.SUPPRESS_WARNINGS = False
            rd = self.test_data.get_rd('cancm4_tas')
            metadata = deepcopy(rd.source_metadata)
            metadata['variables'].pop('lat_bnds')
            dim_map = get_dimension_map('tas', metadata)
            self.assertEqual(dim_map['Y']['bounds'], None)

        self.assertWarns(OcgWarning, _run_)
