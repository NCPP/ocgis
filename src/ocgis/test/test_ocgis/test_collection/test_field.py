import datetime
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from mock import Mock
from nose.plugins.skip import SkipTest
from ocgis import RequestDataset, vm, DimensionMap
from ocgis import constants
from ocgis import netcdftime
from ocgis.base import get_variable_names, atleast_ncver
from ocgis.collection.field import Field, get_name_mapping
from ocgis.collection.spatial import SpatialCollection
from ocgis.constants import HeaderName, KeywordArgument, DriverKey, DimensionMapKey, DMK, Topology
from ocgis.conv.nc import NcConverter
from ocgis.driver.csv_ import DriverCSV
from ocgis.driver.nc import DriverNetcdf
from ocgis.driver.vector import DriverVector
from ocgis.spatial.base import create_spatial_mask_variable
from ocgis.spatial.geom_cabinet import GeomCabinetIterator
from ocgis.spatial.grid import Grid
from ocgis.test.base import attr, AbstractTestInterface, create_gridxy_global, create_exact_field
from ocgis.util.helpers import reduce_multiply
from ocgis.variable.base import Variable
from ocgis.variable.crs import CoordinateReferenceSystem, WGS84, Spherical
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.temporal import TemporalVariable
from ocgis.vmachine.mpi import MPI_SIZE, MPI_RANK, MPI_COMM
from shapely.geometry import Point
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


class TestField(AbstractTestInterface):
    def get_ocgfield(self, *args, **kwargs):
        return Field(*args, **kwargs)

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
                'x': {'variable': lon.name, DimensionMapKey.DIMENSION: [lon.dimensions[0].name]},
                'y': {'variable': lat.name, DimensionMapKey.DIMENSION: [lat.dimensions[0].name]}}
        field = Field(variables=[t, lon, lat, tas, garbage1, time_related], dimension_map=dmap, is_data=tas.name)
        return field

    def test_init(self):
        field = self.get_ocgfield()
        self.assertIsInstance(field, Field)

        # Test unique identifier.
        field = Field(uid=4)
        self.assertEqual(field.uid, 4)

        # Test with a coordinate system and geometry.
        desired_crs = WGS84()
        geom = GeometryVariable(name='geom', value=[Point(1, 2)], dimensions='geom')
        field = Field(crs=desired_crs, geom=geom)
        self.assertEqual(field.crs, desired_crs)

        # Test dimension names are automatically added to dimension map.
        g = GeometryVariable(name='geom', value=[Point(1, 2)], dimensions='the_geom_dim')
        f = Field(geom=g)
        actual = f.dimension_map.get_dimension(DimensionMapKey.GEOM)
        self.assertEqual(actual, ['the_geom_dim'])

        # Test dimension map does not have any entries at initialization.
        actual = Field()
        desired = actual.dimension_map.as_dict()
        self.assertEqual(len(desired), 0)

    def test_system_crs_and_grid_abstraction(self):
        f = Field(grid_abstraction='point')
        grid = self.get_gridxy(with_xy_bounds=True)
        f.add_variable(grid.x)

        crs = CoordinateReferenceSystem(epsg=2136, name='location')
        f.add_variable(crs)
        self.assertIsNone(f.crs)
        f.dimension_map.set_crs(crs)
        f.dimension_map.set_variable('x', grid.x)
        f.dimension_map.set_variable('y', grid.y)
        self.assertEqual(f.grid.crs, crs)

        f.set_geom(f.grid.get_abstraction_geometry())
        self.assertEqual(f.grid.abstraction, 'point')
        self.assertEqual(f.geom.geom_type, 'Point')

    def test_system_dimension_map_formatting(self):
        """Test any formatting of the incoming dimension map by the field."""

        dmap = {'time': {'variable': 'time'}}
        time = TemporalVariable(name='time', value=[1, 2, 3], dimensions='the_time')
        field = Field(time=time, dimension_map=dmap)
        actual = field.dimension_map.get_dimension('time')
        self.assertEqual(actual, ['the_time'])

    def test_system_add_variable(self):
        """Test adding variables from spatial collections."""

        # Create a few separate fields.
        variable_names = tuple(['a', 'b', 'c'])
        fields = [self.get_field(variable_name=v) for v in variable_names]
        # Create spatial collections containing those fields.
        scs = []
        for field in fields:
            sc = SpatialCollection()
            sc.add_field(field, None)
            scs.append(sc)

        # Destination spatial collection to add variables to from source spatial collections.
        grow = scs[0]
        # Loop over source fields.
        for idx in range(1, len(scs)):
            # Loop over child fields and spatial containers in the current source spatial collection.
            for field, container in scs[idx].iter_fields(yield_container=True):
                # TODO: This should be adjusted to allow easier selection with empty fields.
                try:
                    # Case when we have spatial containers.
                    grow_field = grow.get_element(field_name=field.name, container_ugid=container)
                except KeyError:
                    # Case without spatial containers.
                    grow_field = grow.get_element(field.name)
                # Add data variables to the grow field.
                for dv in field.data_variables:
                    grow_field.add_variable(dv.extract(), is_data=True)

        # Assert all variables are present on the grow field.
        actual = grow.get_element()
        self.assertEqual(get_variable_names(actual.data_variables), variable_names)

        # Write the spatial collection using a converter.
        conv = NcConverter([grow], outdir=self.current_dir_output, prefix='out.nc')
        conv.write()

        # Assert all variables are present.
        rd = RequestDataset(conv.path)
        actual = rd.get()
        self.assertEqual(get_variable_names(actual.data_variables), variable_names)

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
        f.dimension_map.set_variable('time', time.name)
        self.assertNumpyAll(f.time.get_value(), time.get_value())
        self.assertEqual(f.time.attrs['axis'], 'T')
        self.assertIsNone(f.time.bounds)
        f.dimension_map.set_variable('time', 'time', bounds=time_bounds.name)
        self.assertNumpyAll(f.time.bounds.get_value(), time_bounds.get_value())
        self.assertIn('other', f.time.parent)

        dims = f.dimension_map.get_dimension('time')
        dims += ['times', 'times_again', 'the_time']
        sub = f.get_field_slice({'time': slice(1, 2)})
        desired = OrderedDict([('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)
        self.assertIsNone(sub.grid)
        sub.dimension_map.set_variable('x', 'xc')
        sub.dimension_map.set_variable('y', 'yc')

        # Test writing to netCDF will load attributes.
        path = self.get_temporary_file_path('foo.nc')
        sub.write(path)
        with self.nc_scope(path) as ds:
            self.assertEqual(ds.variables[x.name].axis, 'X')
            self.assertEqual(ds.variables[y.name].axis, 'Y')

        self.assertEqual(sub.x.attrs['axis'], 'X')
        self.assertEqual(sub.y.attrs['axis'], 'Y')
        self.assertIsInstance(sub.grid, Grid)
        desired = OrderedDict([('time', (1,)), ('time_bounds', (1, 2)), ('other', (1,)), ('xc', (3,)), ('yc', (4,))])
        self.assertEqual(sub.shapes, desired)

        # Test a subset.
        bbox = [1.5, 15, 2.5, 35]
        data = Variable(name='data', value=np.random.rand(3, 4), dimensions=['x', 'y'])
        f2.add_variable(data)
        f2.dimension_map.set_variable('x', 'xc')
        f2.dimension_map.set_variable('y', 'yc')
        bbox = box(*bbox)
        spatial_sub = f2.grid.get_intersects(bbox).parent
        desired = OrderedDict([('time', (3,)), ('time_bounds', (3, 2)), ('other', (3,)), ('xc', (1,)), ('yc', (2,)),
                               ('data', (1, 2)), ])
        self.assertEqual(spatial_sub.shapes, desired)

    def test_system_subsetting(self):
        """Test subsetting operations."""

        field = self.get_ocgfield_example()
        field.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'corners')
        sub = field.time.get_between(datetime.datetime(1, 1, 2, 12, 0),
                                     datetime.datetime(1, 1, 4, 12, 0)).parent
        sub = sub.grid.get_intersects(box(*[35, -45, 55, -15])).parent
        self.assertTrue(sub.grid.is_vectorized)

    def test_crs(self):
        """Test overloading by geometry and grid."""

        field = Field()
        self.assertIsNone(field.crs)

        geom = GeometryVariable(name='geom', value=[Point(1, 2)], dimensions='g', crs=Spherical())
        field = Field(geom=geom)
        self.assertEqual(field.crs, geom.crs)

        grid = self.get_gridxy_global(crs=Spherical())
        field = Field(grid=grid)
        self.assertEqual(field.crs, grid.crs)

        grid = self.get_gridxy_global(crs=Spherical())
        # Grid and field coordinate systems do not match.
        with self.assertRaises(ValueError):
            Field(grid=grid, crs=WGS84())

        geom = GeometryVariable(name='geom', value=[Point(1, 2)], dimensions='g', crs=Spherical())
        with self.assertRaises(ValueError):
            Field(geom=geom, crs=WGS84())

        geom = GeometryVariable(name='geom', value=[Point(1, 2)], dimensions='g')
        grid = self.get_gridxy_global()
        field = Field(geom=geom, grid=grid, crs=WGS84())
        self.assertEqual(field.crs, WGS84())
        self.assertEqual(field.geom.crs, WGS84())
        self.assertEqual(field.grid.crs, WGS84())

        g = self.get_gridxy_global()
        f = Field(grid=g, crs=Spherical())
        self.assertIn('standard_name', f.grid.x.attrs)
        self.assertIn('standard_name', f.grid.y.attrs)

    def test_decode(self):
        f = Field()
        f.decode()
        with self.assertRaises(KeyError):
            _ = f['name']
        self.assertIsNone(f.name)

    def test_dimensions(self):
        crs = CoordinateReferenceSystem(epsg=2136)
        field = Field(variables=[crs])
        self.assertEqual(len(field.dimensions), 0)

    @attr('data')
    def test_from_records(self):
        gci = GeomCabinetIterator(path=self.path_state_boundaries)
        actual = Field.from_records(gci, data_model='NETCDF3_CLASSIC')
        desired = {'UGID': np.int32,
                   'ID': np.int32}
        for v in desired.keys():
            self.assertEqual(actual[v].get_value().dtype, desired[v])

    def test_get_by_tag(self):
        v1 = Variable(name='tas')
        v2 = Variable(name='tasmax')
        v3 = Variable(name='tasmin')
        tags = {'avg': ['tas'], 'other': ['tasmax', 'tasmin']}
        field = Field(variables=[v1, v2, v3], tags=tags)
        t = field.get_by_tag('other')
        self.assertAsSetEqual([ii.name for ii in t], tags['other'])

    def test_get_name_mapping(self):
        dimension_map = {'crs': {'variable': 'latitude_longitude'}, 'level': {'variable': None, 'dimension': []},
                         'time': {'variable': u'time', 'attrs': {'axis': 'T'}, 'bounds': u'time_bnds',
                                  'dimension': [u'time']}, 'driver': 'netcdf-cf', 'spatial_mask': {'variable': None},
                         'groups': {}, 'realization': {'variable': None, 'dimension': []},
                         'y': {'variable': u'lat', 'attrs': {}, 'bounds': u'lat_vertices', 'dimension': [u'rlat']},
                         'x': {'variable': u'lon', 'attrs': {}, 'bounds': u'lon_vertices', 'dimension': [u'rlon']}}
        dimension_map = DimensionMap.from_dict(dimension_map)

        actual = get_name_mapping(dimension_map)
        desired = {'y': [u'rlat'], 'x': [u'rlon'], 'time': [u'time']}
        self.assertEqual(actual, desired)

    def test_grid(self):
        # Test mask variable information is propagated through property.
        grid = self.get_gridxy(with_xy_bounds=True)
        self.assertTrue(grid.is_vectorized)
        self.assertTrue(grid.has_bounds)
        np.random.seed(1)
        value = np.random.rand(*grid.shape)
        select = value > 0.4
        mask_var = create_spatial_mask_variable('nonstandard', select, grid.dimensions)
        grid.set_mask(mask_var)
        field = Field(grid=grid)
        self.assertTrue(field.grid.has_bounds)
        self.assertEqual(field.dimension_map.get_spatial_mask(), mask_var.name)
        self.assertNumpyAll(field.grid.get_mask(), mask_var.get_mask())

        # Test dimension map bounds are updated appropriately.
        dim = Dimension('count', 2)
        x = Variable(name='x', value=[1., 2.], dimensions=dim)
        y = Variable(name='y', value=[1., 2.], dimensions=dim)
        xb = Variable(name='xb', value=[[0., 1.5], [1.5, 2.5]], dimensions=[dim, 'bounds'])
        yb = Variable(name='yb', value=[[0., 1.5], [1.5, 2.5]], dimensions=[dim, 'bounds'])
        variables = [x, y, xb, yb]
        dmap = DimensionMap()
        dmap.set_variable(DMK.X, x, bounds=xb)
        dmap.set_variable(DMK.Y, y, bounds=yb)
        f = Field(dimension_map=dmap, variables=variables)
        self.assertTrue(f.grid.has_bounds)

    def test_iter(self):
        field = self.get_ocgfield_example()
        field.set_geom(field.grid.get_abstraction_geometry())
        field.geom.create_ugid(HeaderName.ID_GEOMETRY)

        geom2 = field.geom.deepcopy()
        geom2.set_name('geom2')
        geom2 = geom2.extract()
        field.add_variable(geom2)
        self.assertIn(geom2.name, field)

        desired_tas_sum = field['tas'].get_value().sum()

        keywords = {'standardize': [True, False], 'melted': [False, True],
                    KeywordArgument.DRIVER: [DriverKey.CSV, None]}

        for k in self.iter_product_keywords(keywords):
            try:
                actual_tas_sum = 0.0
                for ctr, (geom, data) in enumerate(field.iter(standardize=k.standardize, melted=k.melted,
                                                              driver=k.driver)):
                    self.assertNotIn(geom2.name, data)
                    self.assertIsInstance(geom, BaseGeometry)
                    if k.standardize and k.driver is None:
                        try:
                            self.assertIsInstance(data['LB_TIME'], datetime.datetime)
                            self.assertIsInstance(data['UB_TIME'], datetime.datetime)
                        except AssertionError:
                            self.assertIsInstance(data['LB_TIME'], netcdftime.datetime)
                            self.assertIsInstance(data['UB_TIME'], netcdftime.datetime)
                    if k.melted:
                        data_value = data[HeaderName.VALUE]
                        actual_tas_sum += data_value
                        self.assertIn(HeaderName.VARIABLE, data)
                    if k.standardize:
                        self.assertIn(HeaderName.ID_GEOMETRY, data)
                        if not k.melted:
                            self.assertIn(HeaderName.TEMPORAL, data)
                            for tb in HeaderName.TEMPORAL_BOUNDS:
                                self.assertIn(tb, data)
                    else:
                        self.assertNotIn(HeaderName.ID_GEOMETRY, data)
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
        field = Field(time=time, is_data=data, variables=data)

        itr = field.iter(allow_masked=True)
        actual = list(itr)
        self.assertIsNone(actual[1][1][data.name])

    def test_iter_two_data_variables(self):
        """Test iteration with two data variables."""

        field = self.get_ocgfield_example()

        tas2 = field['tas'].deepcopy()
        tas2.set_name('tas2')
        tas2 = tas2.extract()

        field.add_variable(tas2, is_data=True)
        self.assertIsNotNone(list(field.iter(melted=True)))

    def test_set_geom(self):
        f = Field()
        self.assertIsNone(f.crs)

        g = GeometryVariable(value=[Point(1, 2)], dimensions='geom', crs=Spherical())

        f.set_geom(g)

    def test_set_x(self):
        f = Field()
        var = Variable('x', value=[1, 2], dimensions='xdim')
        f.set_x(var, 'xdim')

        var2 = Variable('x2', value=[3, 4], dimensions='xdim2')
        f.set_x(var2, 'xdim2')

        self.assertNotIn(var.name, f)

        f.set_x(None, None)
        self.assertEqual(len(f), 0)
        self.assertIsNone(f.x)

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
            f = Field(variables=[var, bounds_var], dimension_map=dimension_map)
            self.assertTrue(len(f.dimension_map._storage) == 1)
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

    @attr('xarray')
    def test_to_xarray(self):
        grid = create_gridxy_global(crs=Spherical())
        field = create_exact_field(grid, 'foo', ntime=3)
        field.attrs['i_am_global'] = 'confirm'
        field.grid.abstraction = Topology.POINT
        field.set_abstraction_geom()
        field.time.set_extrapolated_bounds('time_bounds', 'bounds')
        xr = field.to_xarray()
        self.assertEqual(xr.attrs['i_am_global'], 'confirm')
        self.assertGreater(len(xr.coords), 0)

    def test_update_crs(self):
        # Test copying allows the CRS to be updated on the copy w/out changing the source CRS.
        desired = Spherical()
        gvar = GeometryVariable(value=[Point(1, 2)], name='geom', dimensions='geom')
        field = Field(crs=desired, geom=gvar)
        cfield = field.copy()
        self.assertEqual(cfield.crs, desired)
        new_crs = CoordinateReferenceSystem(name='i_am_new', epsg=4326)
        cfield.update_crs(new_crs)
        self.assertEqual(field.crs, desired)

        # Test geometry crs update is called.
        mfield = Mock(Field)
        mfield.is_empty = False
        mfield.dimension_map = Mock(DimensionMap)
        mfield.grid = Mock(Grid)
        mfield.geom = Mock(GeometryVariable)
        mcrs = Mock(Spherical)
        Field.update_crs(mfield, mcrs)
        mfield.grid.update_crs.assert_called_once_with(mcrs, from_crs=mfield.crs)
        mfield.geom.update_crs.assert_called_once_with(mcrs, from_crs=mfield.crs)
        from_crs = Mock(WGS84)
        mfield.grid.update_crs.reset_mock()
        mfield.geom.update_crs.reset_mock()
        Field.update_crs(mfield, mcrs, from_crs=from_crs)
        mfield.grid.update_crs.assert_called_once_with(mcrs, from_crs=from_crs)
        mfield.geom.update_crs.assert_called_once_with(mcrs, from_crs=from_crs)

    def test_write(self):
        # Test writing a basic grid.
        path = self.get_temporary_file_path('foo.nc')
        x = Variable(name='x', value=[1, 2], dimensions='x')
        y = Variable(name='y', value=[3, 4, 5, 6, 7], dimensions='y')
        dmap = {'x': {'variable': 'x'}, 'y': {'variable': 'y'}}
        field = Field(variables=[x, y], dimension_map=dmap)
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
        field = Field(grid=grid)
        self.assertTrue(field.grid.is_vectorized)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        self.assertTrue(field.grid.is_vectorized)
        with self.nc_scope(path) as ds:
            if atleast_ncver("1.4"):
                actual = grid.x.mv()
            else:
                actual = grid.x.v()
            self.assertNumpyAll(ds.variables[grid.x.name][:], actual)
            var = ds.variables[grid.y.name]
            if atleast_ncver("1.4"):
                actual = grid.y.mv()
            else:
                actual = grid.y.v()
            self.assertNumpyAll(var[:], actual)
            self.assertEqual(var.axis, 'Y')
            self.assertIn(grid.crs.name, ds.variables)

        # Test with 2-d x and y arrays.
        grid = self.get_gridxy(with_2d_variables=True)
        field = Field(grid=grid)
        path = self.get_temporary_file_path('out.nc')
        field.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            var = ds.variables['y']
            if atleast_ncver("1.4"):
                actual = grid.y.mv()
            else:
                actual = grid.y.v()
            self.assertNumpyAll(var[:], actual)

        # Test writing a vectorized grid with corners.
        grid = self.get_gridxy()
        field = Field(grid=grid)
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
        """Test writing by selective rank."""

        if MPI_SIZE != 3 and MPI_SIZE != 1:
            raise SkipTest('MPI_SIZE != 1 or 3')

        ranks = list(range(MPI_SIZE))

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

                with vm.scoped('field write by rank', [base_rank]):
                    if not vm.is_null:
                        geom = GeometryVariable(value=[Point(1, 2), Point(3, 4)], name='geom', dimensions='geom')
                        data = Variable(name='data', value=[10, 20], dimensions='geom')
                        field = Field(geom=geom)
                        field.add_variable(data, is_data=True)
                        self.assertFalse(os.path.isdir(path))
                        field.write(path, driver=driver)
                        self.assertFalse(os.path.isdir(path))

                        rd = RequestDataset(path, driver=driver)
                        in_field = rd.get()
                        self.assertEqual(in_field['data'].dimensions[0].size, 2)
                MPI_COMM.Barrier()
        MPI_COMM.Barrier()
