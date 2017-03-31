import datetime
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from nose.plugins.skip import SkipTest
from shapely.geometry import Point
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from ocgis import RequestDataset
from ocgis import constants
from ocgis.collection.field import OcgField
from ocgis.constants import HeaderNames, KeywordArguments, DriverKeys
from ocgis.driver.csv_ import DriverCSV
from ocgis.driver.nc import DriverNetcdf
from ocgis.driver.vector import DriverVector
from ocgis.spatial.grid import GridXY
from ocgis.test.base import attr, AbstractTestInterface
from ocgis.util.helpers import reduce_multiply
from ocgis.variable.base import Variable
from ocgis.variable.crs import CoordinateReferenceSystem, WGS84, Spherical
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.temporal import TemporalVariable
from ocgis.vm.mpi import MPI_SIZE, MPI_RANK, MPI_COMM


class TestOcgField(AbstractTestInterface):
    def get_ocgfield(self, *args, **kwargs):
        return OcgField(*args, **kwargs)

    def get_ocgfield_example(self):
        dtime = Dimension(name='time')
        t = TemporalVariable(value=[1, 2, 3, 4], name='the_time', dimensions=dtime, dtype=float)
        t.set_extrapolated_bounds('the_time_bounds', 'bounds')
        lon = Variable(value=[30., 40., 50., 60.], name='longitude', dimensions='lon')
        lat = Variable(value=[-10., -20., -30., -40., -50.], name='latitude', dimensions='lat')
        tas_shape = [t.shape[0], lat.shape[0], lon.shape[0]]
        tas = Variable(value=np.arange(reduce_multiply(tas_shape)).reshape(*tas_shape),
                       dimensions=(dtime, 'lat', 'lon'), name='tas')
        time_related = Variable(value=[7, 8, 9, 10], name='time_related', dimensions=dtime)
        garbage1 = Variable(value=[66, 67, 68], dimensions='three', name='garbage1')
        dmap = {'time': {'variable': t.name},
                'x': {'variable': lon.name, 'names': [lon.dimensions[0].name]},
                'y': {'variable': lat.name, 'names': [lat.dimensions[0].name]}}
        field = OcgField(variables=[t, lon, lat, tas, garbage1, time_related], dimension_map=dmap, is_data=tas.name)
        return field

    def test_init(self):
        field = self.get_ocgfield()
        self.assertIsInstance(field, OcgField)

        # Test unique identifier.
        field = OcgField(uid=4)
        self.assertEqual(field.uid, 4)

        # Test with a coordinate system and geometry.
        desired_crs = WGS84()
        geom = GeometryVariable(name='geom', value=[Point(1, 2)], dimensions='geom')
        field = OcgField(crs=desired_crs, geom=geom)
        self.assertEqual(field.crs, desired_crs)

        # Test geometry coordinate system is not used. This must be set explicitly on the field.
        crs = CoordinateReferenceSystem(epsg=2136)
        g = GeometryVariable(name='foo', value=[Point(1, 2)], dimensions='geom', crs=crs)
        f = OcgField(geom=g)
        self.assertIsNone(f.crs)

    def test_system_crs_and_grid_abstraction(self):
        f = OcgField(grid_abstraction='point')
        grid = self.get_gridxy(with_xy_bounds=True)
        f.add_variable(grid.x)

        crs = CoordinateReferenceSystem(epsg=2136, name='location')
        f.add_variable(crs)
        self.assertIsNone(f.crs)
        f.dimension_map['crs']['variable'] = crs.name
        f.dimension_map['x']['variable'] = grid.x.name
        f.dimension_map['y']['variable'] = grid.y.name
        self.assertEqual(f.grid.crs, crs)

        f.set_geom(f.grid.get_abstraction_geometry())
        self.assertEqual(f.grid.abstraction, 'point')
        self.assertEqual(f.geom.geom_type, 'Point')

    def test_system_dimension_map_formatting(self):
        """Test any formatting of the incoming dimension map by the field."""

        dmap = {'time': {'variable': 'time'}}
        time = TemporalVariable(name='time', value=[1, 2, 3], dimensions='the_time')
        field = OcgField(time=time, dimension_map=dmap)
        self.assertEqual(field.dimension_map['time']['names'], ['the_time'])

    def test_system_properties(self):
        """Test field properties."""

        time = TemporalVariable(value=[20, 30, 40], dimensions=['the_time'], dtype=float, name='time')
        time_bounds = TemporalVariable(value=[[15, 25], [25, 35], [35, 45]], dimensions=['times', 'bounds'],
                                       dtype=float, name='time_bounds')
        other = Variable(value=[44, 55, 66], name='other', dimensions=['times_again'])
        x = Variable(value=[1, 2, 3], name='xc', dimensions=['x'])
        y = Variable(value=[10, 20, 30, 40], name='yc', dimensions=['y'])

        crs = CoordinateReferenceSystem(epsg=2136)
        f = self.get_ocgfield(variables=[time, time_bounds, other, x, y])
        f2 = deepcopy(f)

        self.assertIsNone(f.realization)
        self.assertIsNone(f.time)
        f.dimension_map['time']['variable'] = time.name
        self.assertNumpyAll(f.time.get_value(), time.get_value())
        self.assertEqual(f.time.attrs['axis'], 'T')
        self.assertIsNone(f.time.bounds)
        f.dimension_map['time']['bounds'] = time_bounds.name
        self.assertNumpyAll(f.time.bounds.get_value(), time_bounds.get_value())
        self.assertIn('other', f.time.parent)

        f.dimension_map['time']['names'] += ['times', 'times_again', 'the_time']
        sub = f.get_field_slice({'time': slice(1, 2)})
        desired = OrderedDict([('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)
        self.assertIsNone(sub.grid)
        sub.dimension_map['x']['variable'] = 'xc'
        sub.dimension_map['y']['variable'] = 'yc'

        # Test writing to netCDF will load attributes.
        path = self.get_temporary_file_path('foo.nc')
        sub.write(path)
        with self.nc_scope(path) as ds:
            self.assertEqual(ds.variables[x.name].axis, 'X')
            self.assertEqual(ds.variables[y.name].axis, 'Y')

        self.assertEqual(sub.x.attrs['axis'], 'X')
        self.assertEqual(sub.y.attrs['axis'], 'Y')
        self.assertIsInstance(sub.grid, GridXY)
        desired = OrderedDict([('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)

        # Test a subset.
        bbox = [1.5, 15, 2.5, 35]
        data = Variable(name='data', value=np.random.rand(3, 4), dimensions=['x', 'y'])
        f2.add_variable(data)
        f2.dimension_map['x']['variable'] = 'xc'
        f2.dimension_map['y']['variable'] = 'yc'
        bbox = box(*bbox)
        spatial_sub = f2.grid.get_intersects(bbox).parent
        desired = OrderedDict([('time', (3,)), ('time_bounds', (3, 2)), ('other', (3,)), ('xc', (1,)), ('yc', (2,)),
                               ('data', (1, 2)), ('ocgis_spatial_mask', (2, 1))])
        self.assertEqual(spatial_sub.shapes, desired)

        # path = self.get_temporary_file_path('foo.nc')
        # spatial_sub.write_netcdf(psath)
        # self.ncdump(path)

    def test_system_subsetting(self):
        """Test subsetting operations."""

        field = self.get_ocgfield_example()
        field.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'corners')
        sub = field.time.get_between(datetime.datetime(1, 1, 2, 12, 0),
                                     datetime.datetime(1, 1, 4, 12, 0)).parent
        sub = sub.grid.get_intersects(box(*[35, -45, 55, -15])).parent
        self.assertTrue(sub.grid.is_vectorized)

    def test_dimensions(self):
        crs = CoordinateReferenceSystem(epsg=2136)
        field = OcgField(variables=[crs])
        self.assertEqual(len(field.dimensions), 0)

    def test_get_by_tag(self):
        v1 = Variable(name='tas')
        v2 = Variable(name='tasmax')
        v3 = Variable(name='tasmin')
        tags = {'avg': ['tas'], 'other': ['tasmax', 'tasmin']}
        field = OcgField(variables=[v1, v2, v3], tags=tags)
        t = field.get_by_tag('other')
        self.assertAsSetEqual([ii.name for ii in t], tags['other'])

    def test_iter(self):
        field = self.get_ocgfield_example()
        field.set_geom(field.grid.get_abstraction_geometry())
        field.geom.create_ugid(HeaderNames.ID_GEOMETRY)

        geom2 = field.geom.deepcopy()
        geom2.set_name('geom2')
        geom2.extract()
        field.add_variable(geom2)
        self.assertIn(geom2.name, field)

        desired_tas_sum = field['tas'].get_value().sum()

        keywords = {'standardize': [True, False], 'melted': [False, True],
                    KeywordArguments.DRIVER: [DriverKeys.CSV, None]}

        for k in self.iter_product_keywords(keywords):
            try:
                actual_tas_sum = 0.0
                for ctr, (geom, data) in enumerate(field.iter(standardize=k.standardize, melted=k.melted,
                                                              driver=k.driver)):
                    self.assertNotIn(geom2.name, data)
                    self.assertIsInstance(geom, BaseGeometry)
                    if k.standardize and k.driver is None:
                        self.assertIsInstance(data['LB_TIME'], datetime.datetime)
                        self.assertIsInstance(data['UB_TIME'], datetime.datetime)
                    if k.melted:
                        data_value = data[HeaderNames.VALUE]
                        actual_tas_sum += data_value
                        self.assertIn(HeaderNames.VARIABLE, data)
                    if k.standardize:
                        self.assertIn(HeaderNames.ID_GEOMETRY, data)
                        if not k.melted:
                            self.assertIn(HeaderNames.TEMPORAL, data)
                            for tb in HeaderNames.TEMPORAL_BOUNDS:
                                self.assertIn(tb, data)
                    else:
                        self.assertNotIn(HeaderNames.ID_GEOMETRY, data)
                        if not k.melted:
                            self.assertIn(field.time.name, data)
                if k.melted:
                    self.assertEqual(actual_tas_sum, desired_tas_sum)
            except ValueError:
                self.assertTrue(k.melted)
                self.assertFalse(k.standardize)
                continue

    def test_iter_masking_and_driver(self):
        """Test mask is set to None."""

        time = TemporalVariable(value=[3, 4, 5], dimensions='time')
        data = Variable(value=[7, 8, 9], name='data', mask=[False, True, False], dimensions='time')
        field = OcgField(time=time, is_data=data, variables=data)

        itr = field.iter(allow_masked=True)
        actual = list(itr)
        self.assertIsNone(actual[1][1][data.name])

    def test_iter_two_data_variables(self):
        """Test iteration with two data variables."""

        field = self.get_ocgfield_example()

        tas2 = field['tas'].deepcopy()
        tas2.set_name('tas2')
        tas2.extract()

        field.add_variable(tas2, is_data=True)
        self.assertIsNotNone(list(field.iter(melted=True)))

    def test_time(self):
        units = [None, 'days since 2012-1-1']
        calendar = [None, '365_day']
        value = [10, 20]
        bounds = [[5, 15], [15, 25]]
        variable_type = [Variable, TemporalVariable]
        bounds_variable_type = [Variable, TemporalVariable]

        keywords = dict(units=units, calendar=calendar, variable_type=variable_type,
                        bounds_variable_type=bounds_variable_type)

        for k in self.iter_product_keywords(keywords):
            attrs = {'units': k.units, 'calendar': k.calendar}
            dimension_map = {'time': {'variable': 'time', 'bounds': 'time_bnds', 'attrs': attrs}}
            var = k.variable_type(name='time', value=value, attrs=attrs, dimensions=['one'])
            bounds_var = k.bounds_variable_type(name='time_bnds', value=bounds, dimensions=['one', 'two'])
            f = OcgField(variables=[var, bounds_var], dimension_map=dimension_map)
            self.assertTrue(len(f.dimension_map) > 1)
            self.assertTrue(f.time.has_bounds)
            self.assertIsInstance(f.time, TemporalVariable)
            self.assertIsInstance(f.time.bounds, TemporalVariable)
            self.assertEqual(f.time.value_datetime.shape, (2,))
            self.assertEqual(f.time.bounds.value_datetime.shape, (2, 2))
            if k.units is None:
                desired = constants.DEFAULT_TEMPORAL_UNITS
            else:
                desired = k.units
            self.assertEqual(f.time.units, desired)
            if k.calendar is None:
                desired = constants.DEFAULT_TEMPORAL_CALENDAR
            else:
                desired = k.calendar
            self.assertEqual(f.time.calendar, desired)
            self.assertEqual(f.time.bounds.calendar, desired)

    def test_update_crs(self):
        # Test copying allows the CRS to be updated on the copy w/out changing the source CRS.
        desired = Spherical()
        gvar = GeometryVariable(value=[Point(1, 2)], name='geom', dimensions='geom')
        field = OcgField(crs=desired, geom=gvar)
        cfield = field.copy()
        self.assertEqual(cfield.crs, desired)
        new_crs = CoordinateReferenceSystem(name='i_am_new', epsg=4326)
        cfield.update_crs(new_crs)
        self.assertEqual(field.crs, desired)

    def test_write(self):
        # Test writing a basic grid.
        path = self.get_temporary_file_path('foo.nc')
        x = Variable(name='x', value=[1, 2], dimensions='x')
        y = Variable(name='y', value=[3, 4, 5, 6, 7], dimensions='y')
        dmap = {'x': {'variable': 'x'}, 'y': {'variable': 'y'}}
        field = OcgField(variables=[x, y], dimension_map=dmap)
        desired_value_stacked = field.grid.get_value_stacked()

        self.assertEqual(field.grid.parent['x'].get_value().shape, (2,))
        self.assertTrue(field.grid.is_vectorized)

        field.write(path)
        out_field = RequestDataset(path).get()
        self.assertTrue(out_field.grid.is_vectorized)
        actual_value_stacked = out_field.grid.get_value_stacked()
        self.assertNumpyAll(actual_value_stacked, desired_value_stacked)

        # Test another grid.
        grid = self.get_gridxy(crs=WGS84())
        self.assertTrue(grid.is_vectorized)
        field = OcgField(grid=grid)
        self.assertTrue(field.grid.is_vectorized)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        self.assertTrue(field.grid.is_vectorized)
        with self.nc_scope(path) as ds:
            self.assertNumpyAll(ds.variables[grid.x.name][:], grid.x.get_value())
            var = ds.variables[grid.y.name]
            self.assertNumpyAll(var[:], grid.y.get_value())
            self.assertEqual(var.axis, 'Y')
            self.assertIn(grid.crs.name, ds.variables)

        # Test with 2-d x and y arrays.
        grid = self.get_gridxy(with_2d_variables=True)
        field = OcgField(grid=grid)
        path = self.get_temporary_file_path('out.nc')
        field.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            self.assertNumpyAll(var[:], grid.y.get_value())

        # Test writing a vectorized grid with corners.
        grid = self.get_gridxy()
        field = OcgField(grid=grid)
        self.assertIsNotNone(field.grid.dimensions)
        self.assertFalse(field.grid.has_bounds)
        field.grid.set_extrapolated_bounds('xbnds', 'ybnds', 'corners')
        self.assertTrue(field.grid.is_vectorized)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path, 'r') as ds:
            self.assertEqual(['ydim'], [d for d in ds.variables['y'].dimensions])
            self.assertEqual(['xdim'], [d for d in ds.variables['x'].dimensions])

    @attr('mpi')
    def test_write_parallel(self):
        # Test writing by selective rank.
        if MPI_SIZE != 3 and MPI_SIZE != 1:
            raise SkipTest('serial or mpi-3 only')

        ranks = range(MPI_SIZE)

        for base_rank in ranks:
            for driver in [
                DriverCSV,
                DriverVector,
                DriverNetcdf
            ]:

                if MPI_RANK == 0:
                    path = self.get_temporary_file_path('{}-{}.{}'.format(driver.key, base_rank,
                                                                          driver.common_extension))
                else:
                    path = None
                path = MPI_COMM.bcast(path)

                if MPI_RANK == base_rank:
                    geom = GeometryVariable(value=[Point(1, 2), Point(3, 4)], name='geom', dimensions='geom')
                    data = Variable(name='data', value=[10, 20], dimensions='geom')
                    field = OcgField(geom=geom)
                    field.add_variable(data, is_data=data)
                    self.assertFalse(os.path.isdir(path))
                    field.write(path, driver=driver, ranks_to_write=[MPI_RANK])
                    self.assertFalse(os.path.isdir(path))

                    rd = RequestDataset(path, driver=driver)
                    in_field = rd.get()
                    self.assertEqual(in_field['data'].dimensions[0].size, 2)
                MPI_COMM.Barrier()
        MPI_COMM.Barrier()
