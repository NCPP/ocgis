import pickle
from collections import OrderedDict
from copy import deepcopy

import fiona
import numpy as np
from mock import mock
from shapely.geometry.geo import shape

from ocgis import GeomCabinet, vm
from ocgis import RequestDataset
from ocgis import env
from ocgis.base import get_variable_names
from ocgis.collection.field import Field
from ocgis.constants import DimensionMapKey, DMK
from ocgis.driver.base import iter_all_group_keys
from ocgis.driver.dimension_map import DimensionMap
from ocgis.driver.nc import DriverNetcdf, DriverNetcdfCF, remove_netcdf_attribute, get_crs_variable
from ocgis.exc import OcgWarning, CannotFormatTimeError, \
    NoDataVariablesFound
from ocgis.spatial.grid import Grid
from ocgis.test.base import TestBase, attr, create_gridxy_global
from ocgis.util.addict import Dict
from ocgis.util.helpers import get_group
from ocgis.variable.base import Variable, ObjectType, VariableCollection, SourcedVariable
from ocgis.variable.crs import WGS84, CoordinateReferenceSystem, CFSpherical, CFRotatedPole
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.temporal import TemporalVariable
from ocgis.vmachine.mpi import MPI_RANK, MPI_COMM, OcgDist, variable_scatter


class Test(TestBase):
    def test_get_crs_variable(self):
        # Test with no linking to data variables.
        metadata = {'variables': {'rotated_pole': {'dimensions': (),
                                                   'dtype': np.dtype('S1'),
                                                   'fill_value': 'auto',
                                                   'name': u'rotated_pole',
                                                   'dtype_packed': None,
                                                   'attrs': OrderedDict([(
                                                       u'grid_mapping_name',
                                                       u'rotated_latitude_longitude'),
                                                       (
                                                           u'grid_north_pole_latitude',
                                                           90.0),
                                                       (
                                                           u'grid_north_pole_longitude',
                                                           360.0)]),
                                                   'fill_value_packed': None}}}
        var = get_crs_variable(metadata, False)
        self.assertIsInstance(var, CFRotatedPole)

    def test_remove_netcdf_attribute(self):
        path = self.get_temporary_file_path('foo.nc')
        var = Variable(name='test', attrs={'remove_me': 10})
        var.write(path)

        remove_netcdf_attribute(path, var.name, 'remove_me')

        with self.nc_scope(path) as ds:
            actual = ds.variables[var.name]
            self.assertFalse(hasattr(actual, 'remove_me'))


class TestDriverNetcdf(TestBase):
    def test_init(self):
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('a', 2)
        rd = RequestDataset(uri=path, driver='netcdf')
        self.assertIsInstance(rd.driver, DriverNetcdf)
        vc = rd.get_raw_field()
        self.assertEqual(len(vc), 0)

    def test_system_changing_field_name(self):
        path1 = self.get_temporary_file_path('foo1.nc')
        path2 = self.get_temporary_file_path('foo2.nc')

        vc1 = VariableCollection(name='vc1')
        var1 = Variable('var1', value=[1, 2, 3], dimensions='three', parent=vc1)

        vc2 = VariableCollection(name='vc2')
        vc1.add_child(vc2)
        var2 = Variable('var2', value=[4, 5, 6, 7], dimensions='four', parent=vc2)

        vc1.write(path1)

        rd = RequestDataset(path1)
        # rd.inspect()
        nvc = rd.get_raw_field()
        nvc2 = nvc.children['vc2']
        self.assertIsNone(nvc2['var2']._value)
        self.assertEqual(nvc2.name, 'vc2')
        nvc2.set_name('extraordinary')
        self.assertIsNotNone(nvc2['var2'].get_value())
        self.assertEqual(nvc2['var2'].get_value().tolist(), [4, 5, 6, 7])

        nvc.write(path2)
        rd2 = RequestDataset(path2)
        # rd2.inspect()
        n2vc = rd2.get_raw_field()
        self.assertEqual(n2vc.children[nvc2.name].name, nvc2.name)

    def test_get_dist(self):

        def _create_dimensions_(ds, k):
            if k.dim_count > 0:
                ds.createDimension('one', 1)
                if k.dim_count == 2:
                    ds.createDimension('two', 2)

        kwds = dict(dim_count=[0, 1, 2], nested=[False, True])
        for k in self.iter_product_keywords(kwds):
            path = self.get_temporary_file_path('{}.nc'.format(k.dim_count))
            with self.nc_scope(path, 'w') as ds:
                _create_dimensions_(ds, k)
                if k.nested:
                    group1 = ds.createGroup('nest1')
                    _create_dimensions_(group1, k)
                    group2 = group1.createGroup('nest2')
                    _create_dimensions_(group2, k)
                    group3 = group2.createGroup('nest1')
                    _create_dimensions_(group3, k)
                    group3a = group2.createGroup('nest3')
                    _create_dimensions_(group3a, k)
                    group3.createDimension('outlier', 4)
            rd = RequestDataset(uri=path)
            driver = DriverNetcdf(rd)

            actual = driver.get_dist().mapping

            # All dimensions are not distributed.
            for keyseq in iter_all_group_keys(actual[MPI_RANK]):
                group = get_group(actual[MPI_RANK], keyseq)
                for dim in list(group['dimensions'].values()):
                    self.assertFalse(dim.dist)

            if k.dim_count == 0 and k.nested:
                desired = {None: {'variables': {}, 'dimensions': {}, 'groups': {
                    'nest1': {'variables': {}, 'dimensions': {}, 'groups': {
                        'nest2': {'variables': {}, 'dimensions': {}, 'groups': {'nest1': {'variables': {},
                                                                                          'dimensions': {
                                                                                              'outlier': Dimension(
                                                                                                  name='outlier',
                                                                                                  size=4,
                                                                                                  size_current=4,
                                                                                                  dist=False,
                                                                                                  src_idx='auto')},
                                                                                          'groups': {}}}}}}}}}
                self.assertEqual(actual[MPI_RANK], desired)

            if k.dim_count == 2 and k.nested:
                self.assertIsNotNone(driver.metadata_source['groups']['nest1']['groups']['nest2'])
                two_dimensions = [Dimension(name='one', size=1, size_current=1),
                                  Dimension(name='two', size=2, size_current=2)]
                nest1 = {'dimensions': two_dimensions, 'groups': {}}
                template = deepcopy(nest1)
                nest1['groups']['nest2'] = deepcopy(template)
                nest1['groups']['nest2']['groups']['nest1'] = deepcopy(template)
                nest1['groups']['nest2']['groups']['nest3'] = deepcopy(template)
                nest1['groups']['nest2']['groups']['nest1']['dimensions'].append(Dimension('outlier', 4))
                desired = {None: {'dimensions': two_dimensions, 'groups': {'nest1': nest1}}}
                groups_actual = list(iter_all_group_keys((actual[MPI_RANK])))
                groups_desired = list(iter_all_group_keys(desired))
                self.assertEqual(groups_actual, groups_desired)

    @attr('mpi')
    def test_get_dist_default_distribution(self):
        """Test using default distributions defined by drivers."""

        with vm.scoped('write', [0]):
            if not vm.is_null:
                path = self.get_temporary_file_path('foo.nc')
                varx = Variable('x', np.arange(5), dimensions='five', attrs={'axis': 'X'})
                vary = Variable('y', np.arange(7) + 10, dimensions='seven', attrs={'axis': 'Y'})
                vc = VariableCollection(variables=[varx, vary])
                vc.write(path)
            else:
                path = None
        path = MPI_COMM.bcast(path)

        rd = RequestDataset(path)
        dist = rd.driver.dist

        distributed_dimension = dist.get_dimension('seven')
        self.assertTrue(distributed_dimension.dist)

    def test_get_dump_report(self):
        # Test with nested groups.
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.convention = 'CF Free-For-All 0.99'
            ds.createDimension('dim_root', 5)
            var1_root = ds.createVariable('var1_root', float, dimensions=('dim_root',))
            var1_root.who_knew = 'I did not.'

            group1 = ds.createGroup('group1')
            group1.createDimension('dim_group1', 7)
            var1_group1 = group1.createVariable('var1_group1', int, dimensions=('dim_group1',))
            var1_group1.whatever = 'End of the line!'

            group1_group1 = group1.createGroup('group1_group1')
            group1_group1.createDimension('dim_group1_group1', 10)
            var1_group1_group1 = group1_group1.createVariable('bitter_end', float, dimensions=('dim_group1_group1',))
            var1_group1_group1.foo = 70

        rd = RequestDataset(path)
        driver = DriverNetcdf(rd)
        lines = driver.get_dump_report()
        desired = ['OCGIS Driver Key: netcdf {', 'dimensions:', '    dim_root = 5 ;', 'variables:',
                   '    float64 var1_root(dim_root) ;', '      var1_root:who_knew = "I did not." ;', '',
                   '// global attributes:', '    :convention = "CF Free-For-All 0.99" ;', '', 'group: group1 {',
                   '  dimensions:', '      dim_group1 = 7 ;', '  variables:', '      int64 var1_group1(dim_group1) ;',
                   '        var1_group1:whatever = "End of the line!" ;', '', '  group: group1_group1 {',
                   '    dimensions:', '        dim_group1_group1 = 10 ;', '    variables:',
                   '        float64 bitter_end(dim_group1_group1) ;', '          bitter_end:foo = 70 ;',
                   '    } // group: group1_group1', '  } // group: group1', '}']
        self.assertEqual(lines, desired)

    def test_open(self):
        # Test with a multi-file dataset.
        path1 = self.get_temporary_file_path('foo1.nc')
        path2 = self.get_temporary_file_path('foo2.nc')
        for idx, path in enumerate([path1, path2]):
            with self.nc_scope(path, 'w', format='NETCDF4_CLASSIC') as ds:
                ds.createDimension('a', None)
                b = ds.createVariable('b', np.int32, ('a',))
                b[:] = idx
        uri = [path1, path2]
        rd = RequestDataset(uri=uri, driver=DriverNetcdf)
        field = rd.get_raw_field()
        self.assertEqual(field['b'].get_value().tolist(), [0, 1])

    def test_write_variable(self):
        path = self.get_temporary_file_path('foo.nc')
        var = Variable(name='height', value=10.0, dimensions=[])
        var.write(path)

        rd = RequestDataset(path)
        varin = SourcedVariable(name='height', request_dataset=rd)
        self.assertEqual(varin.get_value(), var.get_value())

        # Test mask persists after write.
        v = Variable(name='the_mask', value=[1, 2, 3, 4], mask=[False, True, True, False], dimensions='ephemeral',
                     fill_value=222)
        path = self.get_temporary_file_path('foo.nc')
        v.write(path)
        rd = RequestDataset(path, driver=DriverNetcdf)
        sv = SourcedVariable(name='the_mask', request_dataset=rd)
        self.assertEqual(sv.get_value().tolist(), [1, 222, 222, 4])
        self.assertNumpyAll(sv.get_mask(), v.get_mask())

    @attr('mpi')
    def test_write_variable_collection(self):
        if MPI_RANK == 0:
            path_in = self.get_temporary_file_path('foo.nc')
            path_out = self.get_temporary_file_path('foo_out.nc')
            with self.nc_scope(path_in, 'w') as ds:
                ds.createDimension('seven', 7)
                var = ds.createVariable('var_seven', float, dimensions=('seven',))
                var[:] = np.arange(7, dtype=float) + 10
                var.foo = 'bar'
        else:
            path_in, path_out = [None] * 2
        path_in = MPI_COMM.bcast(path_in)
        path_out = MPI_COMM.bcast(path_out)

        rd = RequestDataset(path_in)
        rd.metadata['dimensions']['seven']['dist'] = True
        driver = DriverNetcdf(rd)
        vc = driver.get_raw_field()
        with vm.scoped_by_emptyable('write', vc):
            if not vm.is_null:
                vc.write(path_out)

        if MPI_RANK == 0:
            self.assertNcEqual(path_in, path_out)

    def test_write_variable_collection_dataset_variable_kwargs(self):
        """Test writing while overloading things like the dataset data model."""

        path_in = self.get_temporary_file_path('foo.nc')
        path_out = self.get_temporary_file_path('foo_out.nc')
        with self.nc_scope(path_in, 'w', format='NETCDF3_CLASSIC') as ds:
            ds.createDimension('seven', 7)
            var = ds.createVariable('var_seven', np.float32, dimensions=('seven',))
            var[:] = np.arange(7, dtype=np.float32) + 10
            var.foo = 'bar'

        rd = RequestDataset(path_in)
        driver = DriverNetcdf(rd)
        vc = driver.get_raw_field()
        vc.write(path_out, dataset_kwargs={'format': 'NETCDF3_CLASSIC'}, variable_kwargs={'zlib': True})

        self.assertNcEqual(path_in, path_out, ignore_attributes={'var_seven': ['_FillValue']})

    @attr('mpi')
    def test_write_variable_collection_object_arrays(self):
        """Test writing variable length arrays in parallel."""

        with vm.scoped('write', [0]):
            if not vm.is_null:
                path_actual = self.get_temporary_file_path('in.nc')
                path_desired = self.get_temporary_file_path('out.nc')

                value = [[1, 3, 5],
                         [7, 9],
                         [11]]
                v = Variable(name='objects', value=value, fill_value=4, dtype=ObjectType(int), dimensions='values')
                v.write(path_desired)
            else:
                v, path_actual, path_desired = [None] * 3
        path_actual = MPI_COMM.bcast(path_actual)
        path_desired = MPI_COMM.bcast(path_desired)

        dest_mpi = OcgDist()
        dest_mpi.create_dimension('values', 3, dist=True)
        dest_mpi.update_dimension_bounds()

        scattered = variable_scatter(v, dest_mpi)
        outvc = VariableCollection(variables=[scattered])

        with vm.scoped_by_emptyable('write', outvc):
            if not vm.is_null:
                outvc.write(path_actual)

        if MPI_RANK == 0:
            self.assertNcEqual(path_actual, path_desired)


class TestDriverNetcdfCF(TestBase):
    @property
    def fixture_rotated_spherical_metadata(self):
        ret = {'dimensions': {u'rlon': {'name': u'rlon', 'isunlimited': False, 'size': 194},
                              u'bnds': {'name': u'bnds', 'isunlimited': False, 'size': 2},
                              u'rlat': {'name': u'rlat', 'isunlimited': False, 'size': 201},
                              u'time': {'name': u'time', 'isunlimited': True, 'size': 1826}},
               'global_attributes': {u'model_id': u'DMI-HIRHAM5', u'rcm_version_id': u'v2', u'project_id': u'CORDEX',
                                     u'driving_experiment_name': u'rcp45',
                                     u'institution': u'Danish Meteorological Institute',
                                     u'driving_experiment': u'ICHEC-EC-EARTH,rcp45,r3i1p1',
                                     u'CDO': u'Climate Data Operators version 1.4.0.1 (http://www.mpimet.mpg.de/cdo)',
                                     u'CDI': u'Climate Data Interface version 1.4.0.1', u'contact': u'obc@dmi.dk',
                                     u'product': u'output', u'CORDEX_domain': u'AFR-44',
                                     u'experiment': u'Scenario experiment with ICHEC-EC-EARTH forcing',
                                     u'frequency': u'day', u'driving_model_ensemble_member': u'r3i1p1',
                                     u'experiment_id': u'rcp45', u'NCO': u'4.0.9',
                                     u'creation_date': u'2014-07-11 23:53:54', u'Conventions': u'CF-1.6',
                                     u'driving_model_id': u'ICHEC-EC-EARTH',
                                     u'tracking_id': u'9f58ce4e-0956-11e4-a2fa-6c626dd8513d', u'institute_id': u'DMI'},
               'variables': {
                   u'time_bnds': {'dimensions': (u'time', u'bnds'), 'dtype': np.dtype('float64'), 'fill_value': 'auto',
                                  'fill_value_packed': None, 'dtype_packed': None,
                                  'attrs': {u'units': u'days since 1949-12-01 00:00:00',
                                            u'calendar': u'proleptic_gregorian'}, 'name': u'time_bnds'},
                   u'tas': {'dimensions': (u'time', u'rlat', u'rlon'), 'dtype': np.dtype('float32'),
                            'fill_value': 1e+20,
                            'fill_value_packed': None, 'dtype_packed': None,
                            'attrs': {u'_FillValue': 1e+20, u'coordinates': u'lon lat height',
                                      u'long_name': u'Near-Surface Air Temperature',
                                      u'standard_name': u'air_temperature', u'cell_methods': u'time: mean',
                                      u'units': u'K', u'missing_value': 1e+20}, 'name': u'tas'},
                   u'rlon': {'dimensions': (u'rlon',), 'dtype': np.dtype('float64'), 'fill_value': 'auto',
                             'fill_value_packed': None, 'dtype_packed': None,
                             'attrs': {u'units': u'degrees', u'long_name': u'longitude in rotated pole grid',
                                       u'standard_name': u'grid_longitude', u'axis': u'X'}, 'name': u'rlon'},
                   u'lon': {'dimensions': (u'rlat', u'rlon'), 'dtype': np.dtype('float64'), 'fill_value': 'auto',
                            'fill_value_packed': None, 'dtype_packed': None,
                            'attrs': {u'units': u'degrees_east', u'_CoordinateAxisType': u'Lon',
                                      u'standard_name': u'longitude', u'long_name': u'longitude'}, 'name': u'lon'},
                   u'height': {'dimensions': (), 'dtype': np.dtype('float64'), 'fill_value': 'auto',
                               'fill_value_packed': None, 'dtype_packed': None,
                               'attrs': {u'units': u'm', u'long_name': u'height', u'standard_name': u'height',
                                         u'axis': u'Z', u'positive': u'up'}, 'name': u'height'},
                   u'rotated_pole': {'dimensions': (), 'dtype': np.dtype('S1'), 'fill_value': 'auto',
                                     'fill_value_packed': None, 'dtype_packed': None,
                                     'attrs': {u'grid_north_pole_latitude': 90.0, u'grid_north_pole_longitude': 360.0,
                                               u'grid_mapping_name': u'rotated_latitude_longitude'},
                                     'name': u'rotated_pole'},
                   u'time': {'dimensions': (u'time',), 'dtype': np.dtype('float64'), 'fill_value': 'auto',
                             'fill_value_packed': None, 'dtype_packed': None,
                             'attrs': {u'units': u'days since 1949-12-01 00:00:00', u'calendar': u'proleptic_gregorian',
                                       u'bounds': u'time_bnds'}, 'name': u'time'},
                   u'lat': {'dimensions': (u'rlat', u'rlon'), 'dtype': np.dtype('float64'), 'fill_value': 'auto',
                            'fill_value_packed': None, 'dtype_packed': None,
                            'attrs': {u'units': u'degrees_north', u'_CoordinateAxisType': u'Lat',
                                      u'standard_name': u'latitude', u'long_name': u'latitude'}, 'name': u'lat'},
                   u'rlat': {'dimensions': (u'rlat',), 'dtype': np.dtype('float64'), 'fill_value': 'auto',
                             'fill_value_packed': None, 'dtype_packed': None,
                             'attrs': {u'units': u'degrees', u'long_name': u'latitude in rotated pole grid',
                                       u'standard_name': u'grid_latitude', u'axis': u'Y'}, 'name': u'rlat'}},
               'groups': {}, 'file_format': 'NETCDF4_CLASSIC'}
        return ret

    def get_drivernetcdf(self, **kwargs):
        path = self.get_drivernetcdf_file_path()
        kwargs['uri'] = path
        rd = RequestDataset(**kwargs)
        d = DriverNetcdfCF(rd)
        return d

    def get_drivernetcdf_file_path(self):
        path = self.get_temporary_file_path('drivernetcdf.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.convention = 'CF-1.6'
            ds.createDimension('time')
            ds.createDimension('x', 5)
            ds.createDimension('bounds', 2)

            vx = ds.createVariable('x', np.float32, dimensions=['time', 'x'])
            vx[:] = np.random.rand(3, 5) * 100
            vx.grid_mapping = 'latitude_longitude'

            crs = ds.createVariable('latitude_longitude', np.int8)
            crs.grid_mapping_name = 'latitude_longitude'

            vt = ds.createVariable('time', np.float32, dimensions=['time'])
            vt.axis = 'T'
            vt.climatology = 'time_bounds'
            vt[:] = np.arange(1, 4)
            vtb = ds.createVariable('time_bounds', np.float32, dimensions=['time', 'bounds'])
            vtb[:] = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]

            group1 = ds.createGroup('group1')
            group1.contact = 'email'
            group1.createDimension('y', 4)
            vy = group1.createVariable('y', np.int16, dimensions=['y'])
            vy.scale_factor = 5.0
            vy.add_offset = 100.0
            vy[:] = np.ma.array([1, 2, 3, 4], mask=[False, True, False, False])
        return path

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
                    for k, v in source.schema['properties'].items():
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

    def test_init(self):
        d = self.get_drivernetcdf()
        self.assertIsInstance(d, DriverNetcdf)

    @attr('data')
    def test_system_cf_data_read(self):
        """Test some basic reading operations."""

        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertIsInstance(field, Field)
        self.assertEqual(rd.variable, 'tas')
        self.assertEqual(field['tas'].units, 'K')
        self.assertEqual(len(field.dimensions), 4)
        self.assertIsNotNone(field.crs)
        self.assertIsInstance(field.time, TemporalVariable)

        # Geometry is not loaded automatically from the grid.
        self.assertIsNone(field.geom)
        field.set_abstraction_geom()
        self.assertIsInstance(field.geom, GeometryVariable)

        # Test overloading units.
        path = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=path, units='celsius')
        field = rd.get()
        self.assertEqual(field['tas'].units, 'celsius')

    @attr('data', 'mpi')
    def test_system_cf_data_write_parallel(self):
        """Test some basic reading operations."""

        if MPI_RANK == 0:
            path_out = self.get_temporary_file_path('foo.nc')
        else:
            path_out = None
        path_out = MPI_COMM.bcast(path_out)

        rd = self.test_data.get_rd('cancm4_tas')
        rd.metadata['dimensions']['lat']['dist'] = True
        rd.metadata['dimensions']['lon']['dist'] = True
        field = rd.get()
        field.write(path_out, dataset_kwargs={'format': rd.metadata['file_format']})

        if MPI_RANK == 0:
            ignore_attributes = {'time_bnds': ['units', 'calendar'], 'lat_bnds': ['units'], 'lon_bnds': ['units'],
                                 'tas': ['grid_mapping']}
            self.assertNcEqual(path_out, rd.uri, ignore_variables=['latitude_longitude'],
                               ignore_attributes=ignore_attributes)

    def test_system_get_field_dimensioned_variables(self):
        """Test data is appropriately tagged to identify dimensioned variables."""

        path = self.get_temporary_file_path('foo.nc')
        time = TemporalVariable(value=[1, 2, 3], dimensions='time')
        x = Variable(name='x', value=[10, 20], dimensions='x')
        y = Variable(name='y', value=[30, 40, 50, 60], dimensions='y')
        data1 = Variable(name='data1', value=np.random.rand(3, 4, 2), dimensions=['time', 'y', 'x'])
        data2 = Variable(name='data2', value=np.random.rand(3, 4, 2), dimensions=['time', 'y', 'x'])
        data3 = Variable(name='data3', value=[11, 12, 13], dimensions=['time'])
        field = Field(time=time, grid=Grid(x, y), variables=[data1, data2, data3])
        field.write(path)

        # Test dimensioned variables are read from a file with appropriate metadata.
        rd = RequestDataset(path)
        self.assertEqual(rd.variable, ('data1', 'data2'))
        read_field = rd.get()
        actual = get_variable_names(read_field.data_variables)
        self.assertEqual(actual, ('data1', 'data2'))

        # Test dimensioned variables are overloaded.
        rd = RequestDataset(path, variable='data2')
        read_field = rd.get()
        actual = get_variable_names(read_field.data_variables)
        self.assertEqual(actual, ('data2',))

    def test_get_data_variable_names(self):
        driver = self.get_drivernetcdf()
        dvars = driver.get_data_variable_names(driver.rd.metadata, driver.rd.dimension_map)
        self.assertEqual(len(dvars), 0)

        # Test a found variable.
        dimension_map = {'time': {'variable': 'the_time', DimensionMapKey.DIMENSION: ['tt', 'ttt']},
                         'x': {'variable': 'xx', DimensionMapKey.DIMENSION: ['xx', 'xxx']},
                         'y': {'variable': 'yy', DimensionMapKey.DIMENSION: ['yy', 'yyy']}}
        metadata = {'variables': {'tas': {'dimensions': ('xx', 'ttt', 'yyy')},
                                  'pr': {'dimensions': ('foo',)}}}
        dimension_map = DimensionMap.from_dict(dimension_map)
        dvars = driver.get_data_variable_names(metadata, dimension_map)
        self.assertEqual(dvars, ('tas',))

        # Test request dataset uses the dimensioned variables.
        driver = self.get_drivernetcdf()
        with self.assertRaises(NoDataVariablesFound):
            assert driver.rd.variable

    def test_create_dimension_map(self):
        d = self.get_drivernetcdf()
        dmap = d.create_dimension_map(d.metadata_source, strict=True)
        desired = {'crs': {'variable': 'latitude_longitude'},
                   'time': {'variable': 'time', 'bounds': 'time_bounds', DimensionMapKey.DIMENSION: ['time'],
                            DimensionMapKey.ATTRS: {'axis': 'T'}},
                   'driver': DriverNetcdfCF.key}
        self.assertEqual(dmap.as_dict(), desired)

        def _run_():
            env.SUPPRESS_WARNINGS = False
            metadata = {'variables': {'x': {'name': 'x',
                                            'attrs': {'axis': 'X', 'bounds': 'x_bounds'},
                                            'dimensions': ('xx',)}},
                        'dimensions': {'xx': {'name': 'xx', 'size': None}}}
            d.create_dimension_map(metadata)

        self.assertWarns(OcgWarning, _run_)

        # Test overloaded dimension map from request dataset is used.
        dm = {'level': {'variable': 'does_not_exist', DimensionMapKey.DIMENSION: []}}
        driver = self.get_drivernetcdf(dimension_map=dm)
        actual = driver.rd.dimension_map
        self.assertEqual(actual.get_variable(DimensionMapKey.LEVEL), 'does_not_exist')
        # The driver dimension map always loads from the data.
        self.assertNotEqual(dm, driver.create_dimension_map(driver.metadata_source))
        actual = driver.get_field().dimension_map
        self.assertEqual(actual.get_variable(DMK.LEVEL), dm['level']['variable'])

        # Test a dimension name is converted to a list.
        dmap = {DimensionMapKey.TIME: {DimensionMapKey.VARIABLE: 'time', DimensionMapKey.DIMENSION: 'time'}}
        d = self.get_drivernetcdf(dimension_map=dmap)
        f = d.get_field()
        actual = f.dimension_map.get_dimension(DimensionMapKey.TIME)
        self.assertEqual(actual, ['time'])

    def test_create_dimension_map_2d_spatial_coordinates(self):
        grid = create_gridxy_global()
        grid.expand()
        path = self.get_temporary_file_path('foo.nc')
        f = Field(grid=grid)
        f.write(path)
        rd = RequestDataset(path)
        field = rd.get()
        sub = field.get_field_slice({'y': 10, 'x': 5})
        self.assertEqual(sub.grid.x.shape, (1, 1))

        actual = f.dimension_map.get_dimension(DimensionMapKey.Y)
        self.assertEqual(actual, ['y'])

        actual = f.dimension_map.get_dimension(DimensionMapKey.X)
        self.assertEqual(actual, ['x'])

    def test_create_dimension_map_no_time_axis(self):
        metadata = {'variables': {'time': {'name': 'time', 'attrs': {}, 'dimensions': ['time']}},
                    'dimensions': {}}
        d = self.get_drivernetcdf()
        dmap = d.create_dimension_map(metadata)
        self.assertEqual(dmap.get_variable(DimensionMapKey.TIME), 'time')

    def test_create_dimension_map_rotated_spherical(self):
        rd = mock.create_autospec(RequestDataset)
        rd._has_assigned_coordinate_system = False
        rd.rotated_pole_priority = False

        driver = DriverNetcdfCF(rd)
        dmap = driver.create_dimension_map(self.fixture_rotated_spherical_metadata)
        self.assertEqual(dmap.get_variable(DMK.X), 'lon')
        self.assertEqual(dmap.get_variable(DMK.Y), 'lat')

    def test_create_dimension_map_with_spatial_mask(self):
        path = self.get_temporary_file_path('foo.nc')
        grid = create_gridxy_global()
        gmask = grid.get_mask(create=True)
        gmask[1, 1] = True
        grid.set_mask(gmask)
        grid.parent.write(path)
        rd = RequestDataset(path)
        driver = DriverNetcdfCF(rd)
        dmap = driver.create_dimension_map(driver.metadata_source)
        self.assertIsNotNone(dmap.get_spatial_mask())
        field = rd.get()
        self.assertEqual(field.grid.get_mask().sum(), 1)

        # Test mask variable is blown away if set to None during a read.
        rd = RequestDataset(path)
        rd.dimension_map.set_spatial_mask(None)
        self.assertIsNone(rd.dimension_map.get_spatial_mask())
        # rd.dimension_map.pprint()
        field = rd.get()
        self.assertIsNone(field.grid.get_mask())

    def test_get_crs(self):
        group_metadata = self.fixture_rotated_spherical_metadata

        rd = mock.create_autospec(RequestDataset)

        keywords = Dict()
        keywords.rpp = [True, False]
        keywords.with_spherical = [True, False]

        desired = Dict()
        desired.rpp[True] = CFRotatedPole
        desired.rpp[False] = CFSpherical

        for k in self.iter_product_keywords(keywords):
            cgm = deepcopy(group_metadata)
            if not k.with_spherical:
                cgm['variables'].pop('lat')
                cgm['variables'].pop('lon')
            rd.rotated_pole_priority = k.rpp
            actual = DriverNetcdfCF(rd).get_crs(cgm)
            if k.with_spherical:
                self.assertIsInstance(actual, desired.rpp[k.rpp])
            else:
                self.assertIsInstance(actual, CFRotatedPole)

    def test_get_dump_report(self):
        d = self.get_drivernetcdf()
        r = d.get_dump_report()
        self.assertGreaterEqual(len(r), 24)

    def test_get_field(self):
        driver = self.get_drivernetcdf()
        field = driver.get_field(format_time=False)
        self.assertEqual(field.driver.key, driver.key)
        self.assertIsInstance(field.time, TemporalVariable)
        with self.assertRaises(CannotFormatTimeError):
            assert field.time.value_datetime
        self.assertIsInstance(field.crs, CoordinateReferenceSystem)

        # Test overloading the coordinate system.
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            v = ds.createVariable('latitude_longitude', np.int)
            v.grid_mapping_name = 'latitude_longitude'

        # First, test the default is found.
        rd = RequestDataset(uri=path)
        driver = DriverNetcdfCF(rd)
        self.assertEqual(driver.get_crs(driver.metadata_source), CFSpherical())
        self.assertEqual(driver.get_field().crs, CFSpherical())

        # Second, test the overloaded CRS is found.
        desired = CoordinateReferenceSystem(epsg=2136)
        rd = RequestDataset(uri=path, crs=desired)
        self.assertEqual(rd.crs, desired)
        driver = DriverNetcdfCF(rd)
        self.assertEqual(driver.get_crs(driver.metadata_source), CFSpherical())
        field = driver.get_field()
        self.assertEqual(field.crs, desired)
        # Test file coordinate system variable is removed.
        self.assertNotIn('latitude_longitude', field)

        # Test the default coordinate system is used when nothing is in the file.
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createVariable('nothing', np.int)
        rd = RequestDataset(uri=path)
        driver = DriverNetcdfCF(rd)
        self.assertEqual(rd.crs, env.DEFAULT_COORDSYS)
        self.assertEqual(driver.get_crs(driver.rd.metadata), env.DEFAULT_COORDSYS)
        self.assertEqual(driver.get_field().crs, env.DEFAULT_COORDSYS)

    def test_get_field_write_target(self):
        # Test coordinate system names are added to attributes of dimensioned variables.
        x = Variable('x', dimensions='x', value=[1])
        y = Variable('y', dimensions='y', value=[2])
        t = Variable('t', dimensions='t', value=[3])
        crs = WGS84()
        d = Variable('data', dimensions=['t', 'y', 'x'], value=[[[1]]])
        grid = Grid(x, y)
        field = Field(grid=grid, time=t, crs=crs)
        field.add_variable(d, is_data=True)

        target = DriverNetcdfCF._get_field_write_target_(field)
        self.assertEqual(target[d.name].attrs['grid_mapping'], crs.name)
        self.assertEqual(field.x.units, 'degrees_east')

        # Test bounds units are removed when writing.
        x = Variable(name='x', value=[1, 2, 3], dtype=float, dimensions='x', units='hours')
        y = Variable(name='y', value=[1, 2, 3], dtype=float, dimensions='x', units='hours')
        grid = Grid(x, y)
        grid.set_extrapolated_bounds('x_bounds', 'y_bounds', 'bounds')
        self.assertEqual(x.bounds.units, x.units)
        self.assertEqual(y.bounds.units, y.units)
        field = Field(grid=grid)
        actual = DriverNetcdfCF._get_field_write_target_(field)
        self.assertEqual(x.bounds.units, x.units)
        self.assertNumpyMayShareMemory(actual[x.name].get_value(), field[x.name].get_value())
        self.assertIsNone(actual[x.name].bounds.units)
        self.assertIsNone(actual[y.name].bounds.units)
        self.assertEqual(x.bounds.units, x.units)
        self.assertEqual(y.bounds.units, y.units)

    def test_metadata_raw(self):
        d = self.get_drivernetcdf()
        metadata = d.metadata_raw
        self.assertIsInstance(metadata, dict)

        desired = metadata.copy()
        pickled = pickle.dumps(metadata)
        unpickled = pickle.loads(pickled)
        self.assertEqual(unpickled, desired)
