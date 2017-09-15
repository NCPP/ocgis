import os

import fiona
import numpy as np
import six
from shapely.geometry import Point
from shapely.ops import cascaded_union

from ocgis import RequestDataset, vm
from ocgis import constants
from ocgis.collection.field import Field
from ocgis.constants import MPIWriteMode, DimensionName
from ocgis.driver.base import AbstractDriver
from ocgis.driver.vector import DriverVector, get_fiona_crs, get_fiona_schema
from ocgis.ops.core import OcgOperations
from ocgis.spatial.geom_cabinet import GeomCabinetIterator
from ocgis.test.base import TestBase, attr, create_exact_field, create_gridxy_global
from ocgis.variable.base import Variable, SourcedVariable
from ocgis.variable.crs import WGS84, CoordinateReferenceSystem, Spherical
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.temporal import TemporalVariable
from ocgis.vmachine.mpi import MPI_RANK, MPI_COMM, MPI_SIZE, OcgDist, variable_collection_scatter, hgather


class TestDriverVector(TestBase):
    def assertOGRFileLength(self, path, desired):
        with fiona.open(path) as source:
            self.assertEqual(len(source), desired)

    def assertPrivateValueIsNone(self, field_like):
        for v in list(field_like.values()):
            if not isinstance(v, CoordinateReferenceSystem):
                self.assertIsNone(v._value)

    def get_driver(self, **kwargs):
        rd = self.get_request_dataset(**kwargs)
        driver = DriverVector(rd)
        return driver

    def get_request_dataset(self, variable=None):
        uri = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        rd = RequestDataset(uri=uri, driver='vector', variable=variable)
        return rd

    def test_init(self):
        self.assertIsInstances(self.get_driver(), (DriverVector, AbstractDriver))

        actual = [constants.OutputFormatName.OCGIS, constants.OutputFormatName.SHAPEFILE]
        self.assertAsSetEqual(actual, DriverVector.output_formats)

    @attr('data')
    def test_system_cf_data(self):
        rd = self.test_data.get_rd('cancm4_tas')
        path = self.get_temporary_file_path('grid.shp')
        field = rd.get()[{'time': slice(3, 6), 'lat': slice(10, 20), 'lon': slice(21, 27)}]
        variable_names = ['time', 'lat', 'lon', 'tas']
        field.set_abstraction_geom()
        field.write(path, driver=DriverVector, variable_names=variable_names)

        read = RequestDataset(path).get()
        self.assertEqual(len(list(read.dimensions.values())[0]), 3 * 10 * 6)
        self.assertEqual(read.crs, Spherical())
        self.assertEqual(rd.get().crs, Spherical())

    @attr('cfunits')
    def test_system_conform_units(self):
        """Test conforming units on data read from shapefile."""

        path = self.get_temporary_file_path('temps.shp')
        gvar = GeometryVariable(value=[Point(1, 2), Point(3, 4)], dimensions='g', name='geom')
        var = Variable(name='temp', value=[10., 20.], dimensions='g')
        field = Field(variables=[gvar, var], geom=gvar, is_data=var)
        field.write(path, driver=DriverVector)

        field = RequestDataset(path, units='celsius', variable='temp', conform_units_to='fahrenheit').get()
        self.assertNumpyAllClose(field['temp'].get_value(), np.array([50., 68.]))

    def test_system_convert_to_geojson(self):
        """GeoJSON conversion does not support update/append write mode."""

        driver = self.get_driver()
        field = driver.get_field()
        path = self.get_temporary_file_path('foo.geojson')
        field.write(path, driver=DriverVector, fiona_driver='GeoJSON')
        # subprocess.check_call(['gedit', path])

    def test_system_merge_geometries_across_shapefiles(self):
        geoms_to_union = []
        state_names = ('Nebraska', 'South Dakota', 'North Dakota')
        gci = GeomCabinetIterator(path=self.path_state_boundaries)
        for row in gci:
            if row['properties']['STATE_NAME'] in state_names:
                geoms_to_union.append(row['geom'])
        self.assertEqual(len(geoms_to_union), 3)
        unioned = cascaded_union(geoms_to_union)

        grid = create_gridxy_global()
        field = create_exact_field(grid, 'data', crs=WGS84())
        original_shape = field.grid.shape
        ops = OcgOperations(dataset=field, geom=unioned)
        ret = ops.execute()
        actual_shape = ret.get_element().grid.shape
        self.assertNotEqual(actual_shape, original_shape)

    @attr('mpi')
    def test_system_parallel_write_ndvariable(self):
        """Test a parallel vector GIS write with a n-dimensional variable."""

        ompi = OcgDist()
        ompi.create_dimension('time', 3)
        ompi.create_dimension('extra', 2)
        ompi.create_dimension('x', 4)
        ompi.create_dimension('y', 7, dist=True)
        ompi.update_dimension_bounds()

        if MPI_RANK == 0:
            path = self.get_temporary_file_path('foo.shp')

            t = TemporalVariable(name='time', value=[1, 2, 3], dtype=float, dimensions='time')
            t.set_extrapolated_bounds('the_time_bounds', 'bounds')

            extra = Variable(name='extra', value=[7, 8], dimensions='extra')

            x = Variable(name='x', value=[9, 10, 11, 12], dimensions='x', dtype=float)
            x.set_extrapolated_bounds('x_bounds', 'bounds')

            # This will have the distributed dimension.
            y = Variable(name='y', value=[13, 14, 15, 16, 17, 18, 19], dimensions='y', dtype=float)
            y.set_extrapolated_bounds('y_bounds', 'bounds')

            data = Variable(name='data', value=np.random.rand(3, 2, 7, 4), dimensions=['time', 'extra', 'y', 'x'])

            dimension_map = {'x': {'variable': 'x', 'bounds': 'x_bounds'},
                             'y': {'variable': 'y', 'bounds': 'y_bounds'},
                             'time': {'variable': 'time', 'bounds': 'the_time_bounds'}}

            vc = Field(variables=[t, extra, x, y, data], dimension_map=dimension_map, is_data='data')
            vc.set_abstraction_geom()
        else:
            path, vc = [None] * 2

        path = MPI_COMM.bcast(path)
        vc = variable_collection_scatter(vc, ompi)
        with vm.scoped_by_emptyable('write', vc):
            if not vm.is_null:
                vc.write(path, driver=DriverVector)
        MPI_COMM.Barrier()

        desired = 168
        rd = RequestDataset(path, driver=DriverVector)
        sizes = MPI_COMM.gather(rd.get().geom.shape[0])
        if MPI_RANK == 0:
            self.assertEqual(sum(sizes), desired)

    @attr('mpi')
    def test_system_with_distributed_dimensions_from_file_shapefile(self):
        """Test a distributed read from file."""

        path = self.path_state_boundaries

        # These are the desired values.
        with vm.scoped('desired data write', [0]):
            if not vm.is_null:
                rd_desired = RequestDataset(uri=path, driver=DriverVector)
                var_desired = SourcedVariable(name='STATE_NAME', request_dataset=rd_desired)
                value_desired = var_desired.get_value().tolist()
                self.assertEqual(len(value_desired), 51)

        rd = RequestDataset(uri=path, driver=DriverVector)
        fvar = SourcedVariable(name='STATE_NAME', request_dataset=rd)
        self.assertEqual(len(rd.driver.dist.get_group()['dimensions']), 1)

        self.assertTrue(fvar.dimensions[0].dist)
        self.assertIsNotNone(fvar.get_value())
        if MPI_SIZE > 1:
            self.assertLessEqual(fvar.shape[0], 26)

        values = MPI_COMM.gather(fvar.get_value())
        if MPI_RANK == 0:
            values = hgather(values)
            self.assertEqual(values.tolist(), value_desired)
        else:
            self.assertIsNone(values)

    def test_system_with_time_data(self):
        """Test writing data with a time dimension."""

        path = self.get_temporary_file_path('what.shp')
        t = TemporalVariable(value=[1.5, 2.5], name='time', dimensions='time')
        geom = GeometryVariable(value=[Point(1, 2), Point(3, 4)], name='geom', dimensions='time')
        field = Field(variables=[t, geom], dimension_map={'time': {'variable': 'time'},
                                                          'geom': {'variable': 'geom'}})
        field.write(path, iter_kwargs={'variable': 'time'}, driver=DriverVector)

        rd = RequestDataset(uri=path)
        field2 = rd.get()

        # netcdftime worthlessness
        poss = [['0001-01-02 12:00:00', '0001-01-03 12:00:00'], ['1-01-02 12:00:00', '1-01-03 12:00:00']]
        actual = field2['TIME'].get_value().tolist()
        res = [p == actual for p in poss]
        self.assertTrue(any(res))

    def test_close(self):
        driver = self.get_driver()
        sci = driver.open(rd=driver.rd)
        driver.close(sci)

    def test_get_crs(self):
        driver = self.get_driver()
        self.assertEqual(WGS84(), driver.get_crs(driver.metadata_source))

    def test_get_data_variable_names(self):
        driver = self.get_driver()
        actual = driver.get_data_variable_names(driver.rd.metadata, driver.rd.dimension_map)
        self.assertEqual(actual, ('UGID', 'STATE_FIPS', 'ID', 'STATE_NAME', 'STATE_ABBR'))

    def test_get_dimensions(self):
        driver = self.get_driver()
        actual = driver.get_dist().mapping[MPI_RANK]
        desired = {None: {
            'variables': {'STATE_FIPS': {'dimensions': ('ocgis_ngeom',)},
                          'STATE_ABBR': {'dimensions': ('ocgis_ngeom',)},
                          'UGID': {'dimensions': ('ocgis_ngeom',)},
                          'ID': {'dimensions': ('ocgis_ngeom',)},
                          'STATE_NAME': {'dimensions': ('ocgis_ngeom',)}},
            'dimensions': {
                'ocgis_ngeom': Dimension(name='ocgis_ngeom', size=51, size_current=51, dist=True, src_idx='auto')},
            'groups': {}}}
        self.assertEqual(actual, desired)

    def test_get_dump_report(self):
        driver = self.get_driver()
        lines = driver.get_dump_report()
        self.assertTrue(len(lines) > 5)

    def test_get_field(self):
        driver = self.get_driver()
        field = driver.get_field()
        self.assertPrivateValueIsNone(field)
        self.assertEqual(len(field), 7)
        self.assertIsInstance(field.geom, GeometryVariable)
        self.assertIsInstance(field.crs, CoordinateReferenceSystem)
        self.assertIsNone(field.time)
        for v in list(field.values()):
            if not isinstance(v, CoordinateReferenceSystem):
                self.assertIsNotNone(v.get_value())

        # Test slicing does not break loading from file.
        field = driver.get_field()
        self.assertPrivateValueIsNone(field)
        sub = field.geom[10, 15, 25].parent
        self.assertPrivateValueIsNone(sub)
        self.assertEqual(len(sub.dimensions[DimensionName.GEOMETRY_DIMENSION]), 3)

    def test_get_raw_field(self):
        driver = self.get_driver()
        vc = driver.get_raw_field()
        self.assertEqual(len(vc), 7)
        for v in list(vc.values()):
            if not isinstance(v, CoordinateReferenceSystem):
                self.assertEqual(len(v.dimensions[0]), 51)
                self.assertIsNone(v._value)

    def test_inspect(self):
        driver = self.get_driver()
        with self.print_scope() as ps:
            driver.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    def test_metadata_source(self):
        driver = self.get_driver()
        m = driver.metadata_source
        self.assertIsInstance(m, dict)
        self.assertIsInstance(m['groups'], dict)
        self.assertTrue(len(m) > 2)
        self.assertIn('variables', m)

    def test_write_variable_collection(self):
        # Attempt to write without a geometry variable.
        v = Variable('a', value=[1, 2], dimensions='bb')
        field = Field(variables=v)
        path = self.get_temporary_file_path('out.shp')
        with self.assertRaises(ValueError):
            field.write(path, driver=DriverVector)

        # Test writing a field with two-dimensional geometry storage.
        value = [Point(1, 2), Point(3, 4), Point(5, 6), Point(6, 7), Point(8, 9), Point(10, 11)]
        gvar = GeometryVariable(value=value, name='points', dimensions='ngeoms')
        gvar.reshape([Dimension('lat', 2), Dimension('lon', 3)])
        var1 = Variable(name='dummy', value=[6, 7, 8], dimensions=['a'])
        var2 = Variable(name='some_lats', value=[41, 41], dimensions=['lat'])
        var3 = Variable(name='some_lons', value=[0, 90, 280], dimensions=['lon'])
        var4 = Variable(name='data', value=np.random.rand(4, 3, 2), dimensions=['time', 'lon', 'lat'])
        field = Field(variables=[var1, var2, var3, var4], geom=gvar, is_data=['data'])
        path = self.get_temporary_file_path('2d.shp')
        field.write(path, iter_kwargs={'followers': ['some_lats', 'some_lons']}, driver=DriverVector)
        read = RequestDataset(uri=path).get()
        self.assertTrue(len(read) > 2)
        self.assertEqual(list(read.keys()),
                         ['data', 'some_lats', 'some_lons', constants.DimensionName.GEOMETRY_DIMENSION])

        # Test writing a subset of the variables.
        path = self.get_temporary_file_path('limited.shp')
        value = [Point(1, 2), Point(3, 4), Point(5, 6)]
        gvar = GeometryVariable(value=value, name='points', dimensions='points')
        var1 = Variable('keep', value=[1, 2, 3], dimensions='points')
        var2 = Variable('remove', value=[4, 5, 6], dimensions='points')
        field = Field(variables=[var1, var2], geom=gvar, is_data=[var1])
        field.write(path, variable_names=['keep'], driver=DriverVector)
        read = RequestDataset(uri=path).get()
        self.assertNotIn('remove', read)

        # Test using append.
        path = self.get_temporary_file_path('limited.shp')
        value = [Point(1, 2), Point(3, 4), Point(5, 6)]
        gvar = GeometryVariable(value=value, name='points', dimensions='points')
        var1 = Variable('keep', value=[1, 2, 3], dimensions='points')
        var2 = Variable('remove', value=[4, 5, 6], dimensions='points')
        field = Field(variables=[var1, var2], geom=gvar, is_data=[var1, var2])
        for idx in range(3):
            sub = field[{'points': idx}]
            if idx == 0:
                write_mode = MPIWriteMode.WRITE
            else:
                write_mode = MPIWriteMode.APPEND
            sub.write(path, write_mode=write_mode, driver=DriverVector)
            self.assertOGRFileLength(path, idx + 1)

    def test_write_variable_collection_different_data_types(self):
        """Test multiple data types are handled by the shapefile write when melted is True."""

        v_int = Variable(name='an_int', value=[1, 2, 3], dtype=int, dimensions='three')
        v_flt = Variable(name='a_float', value=[10., 20., 30.], dtype=float, dimensions='three')
        g = GeometryVariable(name='points', value=[Point(1, 2), Point(3, 4), Point(5, 6)], dimensions='three')
        field = Field(is_data=[v_int, v_flt], geom=g)
        self.assertEqual(len(field.data_variables), 2)
        path = self.get_temporary_file_path('foo.shp')
        field.write(path, driver='vector', iter_kwargs=dict(melted=True))

    @attr('mpi')
    def test_write_variable_collection_parallel(self):
        if MPI_RANK == 0:
            path1 = self.get_temporary_file_path('out1.shp')
            path2 = self.get_temporary_file_path('out2.shp')
        else:
            path1, path2 = [None] * 2
        path1 = MPI_COMM.bcast(path1)
        path2 = MPI_COMM.bcast(path2)

        # Test writing the field to file.
        driver = self.get_driver()
        field = driver.get_field()

        # Only test open file objects on a single processor.
        if MPI_SIZE == 1:
            fiona_crs = get_fiona_crs(field)
            fiona_schema = get_fiona_schema(field.geom.geom_type, six.next(field.iter())[1])
            fobject = fiona.open(path2, mode='w', schema=fiona_schema, crs=fiona_crs, driver='ESRI Shapefile')
        else:
            fobject = None

        for target in [path1, fobject]:
            # Skip the open file object test during a multi-proc test.
            if MPI_SIZE > 1 and target is None:
                continue

            field.write(target, driver=DriverVector)

            if isinstance(target, six.string_types):
                path = path1
            else:
                path = path2
                fobject.close()

            if MPI_RANK == 0:
                with fiona.open(path) as source:
                    self.assertEqual(len(source), 51)
                rd = RequestDataset(uri=path)
                field2 = rd.get()
                for v in list(field.values()):
                    if isinstance(v, CoordinateReferenceSystem):
                        self.assertEqual(v, field2.crs)
                    else:
                        self.assertNumpyAll(v.get_value(), field2[v.name].get_value())
