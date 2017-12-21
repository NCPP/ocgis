import os
from collections import OrderedDict
from copy import deepcopy
from unittest import SkipTest

import numpy as np
from numpy.core.multiarray import ndarray
from numpy.testing.utils import assert_equal
from shapely.geometry import Point

from ocgis import RequestDataset, vm, env
from ocgis.base import get_variable_names
from ocgis.collection.field import Field
from ocgis.constants import HeaderName
from ocgis.exc import VariableInCollectionError, EmptySubsetError, NoUnitsError, PayloadProtectedError, \
    DimensionsRequiredError
from ocgis.ops.core import OcgOperations
from ocgis.test.base import attr, nc_scope, AbstractTestInterface, TestBase
from ocgis.util.units import get_units_object, get_are_units_equal
from ocgis.variable.base import Variable, SourcedVariable, VariableCollection, ObjectType, init_from_source, \
    create_typed_variable_from_data_model
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.vmachine.mpi import MPI_SIZE, MPI_RANK, OcgDist, MPI_COMM, hgather, variable_scatter, variable_gather


class Test(TestBase):
    def test_create_typed_variable_from_data_model(self):
        netcdf_file_format = 'NETCDF3_64BIT_OFFSET'
        string_name = 'int'
        actual = create_typed_variable_from_data_model(string_name, data_model=netcdf_file_format,
                                                       name='test', value=[1, 2, 3, 4], dimensions='dim')
        self.assertEqual(actual.dtype, np.int32)


class TestVariable(AbstractTestInterface):
    def get_boundedvariable_2d(self):
        value = np.array([[2, 2.5],
                          [1, 1.5],
                          [0, 0.5]], dtype=float)
        dims = (Dimension('y', 3), Dimension('x', 2))
        bv = Variable(value=value, name='two_dee', dimensions=dims)
        bv.set_extrapolated_bounds('two_dee_bounds', 'corners')
        return bv

    def get_variable(self, return_original_data=True):
        value = [2, 3, 4, 5, 6, 7]
        time = Dimension('time', size=len(value))
        var = Variable('time_value', value=value, dimensions=time, units='kelvin')
        if return_original_data:
            return time, value, var
        else:
            return var

    def test_init(self):
        # Test an empty variable.
        var = Variable(name='empty')
        self.assertIsNotNone(var.name)
        self.assertIn(var.name, var.parent)
        self.assertEqual(var.shape, tuple())
        self.assertEqual(var.dimensions, tuple())
        self.assertEqual(var.get_value(), None)
        self.assertEqual(var.get_mask(), None)
        self.assertIsNotNone(var.parent)

        # Test with a single dimension.
        var = Variable(dimensions=Dimension('five', 5), name='a_five')
        self.assertEqual(var._dimensions, ('five',))
        self.assertEqual(var.shape, (5,))

        # Test with a single dimension (name only).
        var = Variable(value=[1, 2, 3], dimensions='three', name='a_three')
        self.assertEqual(var.dimensions[0], Dimension('three', 3))

        # Test dimension name only with no shape.
        with self.assertRaises(IndexError):
            Variable(dimensions='what', name='no_shape')

        # Test with a value and no dimensions.
        var = Variable(value=[[2, 3, 4], [4, 5, 6]], name='value', dimensions=['two', 'three'])
        self.assertIsNotNone(var.dimensions)
        self.assertEqual(var.shape, (2, 3))
        self.assertEqual(var.shape, tuple([len(dim) for dim in var.dimensions]))
        self.assertEqual(var.ndim, 2)
        # Variable and dimension shapes must be equal.
        with self.assertRaises(ValueError):
            var.set_value(np.random.rand(10, 11))
        with self.assertRaises(ValueError):
            var.set_dimensions(Dimension('a', None))
        var.set_dimensions([Dimension('aa', 2), Dimension('bb')])
        self.assertEqual(var.dimensions[1].size_current, 3)
        self.assertEqual(var.shape, (2, 3))

        # Test a scalar variable.
        v = Variable(value=2.0, name='scalar', dimensions=[])
        self.assertEqual(v.get_value(), 2.0)
        self.assertIsInstance(v.get_value(), ndarray)
        self.assertEqual(v.shape, tuple())
        self.assertEqual(v.get_value().dtype, np.float)
        self.assertEqual(v.dimensions, tuple())
        self.assertEqual(v.ndim, 0)

        # Test a value with no dimensions.
        with self.assertRaises(ValueError):
            Variable(value=[[1, 2, 3], [4, 5, 6]], name='no dimensions')

        # Test with dimensions only.
        v = Variable(dimensions=[Dimension('a', 3), Dimension('b', 8)], dtype=np.int8, fill_value=2, name='only dims')
        desired = np.zeros((3, 8), dtype=np.int8)
        desired.fill(2)
        self.assertNumpyAll(v.get_value(), desired)

        # Test with an unlimited dimension.
        v = Variable(dimensions=Dimension('unlimited'), name='unhinged')
        with self.assertRaises(ValueError):
            v.get_value()

        # Test value converted to dtype and fill_value.
        value = [4.5, 5.5, 6.5]
        desired = np.array(value, dtype=np.int8)
        var = Variable(value=value, dtype=np.int8, fill_value=4, name='conversion', dimensions='the_dimension')
        self.assertNumpyAll(var.get_value(), desired)
        var._value = None
        var.set_value(np.array(value))
        self.assertNumpyAll(var.get_value(), desired)
        var._value = None
        var.set_value(desired)
        assert_equal(var.get_mask(create=True, check_value=True), [True, False, False])

        # Test with a slice.
        time, value, var = self.get_variable()
        self.assertEqual(var.dimensions, (time,))
        self.assertEqual(id(time), id(var.dimensions[0]))
        self.assertEqual(var.name, 'time_value')
        self.assertEqual(var.shape, (len(value),))
        self.assertNumpyAll(var.get_value(), np.array(value, dtype=var.dtype))
        sub = var[2:4]
        self.assertIsInstance(sub, Variable)
        self.assertEqual(sub.shape, (2,))
        self.assertNumpyAll(sub.get_value(), var.get_value()[2:4])
        self.assertNotEqual(id(var), id(sub))

        # Test conforming data types.
        dtype = np.float32
        fill_value = 33.0
        var = Variable('foo', value=value, dimensions=time, dtype=dtype, fill_value=fill_value)
        self.assertEqual(var.dtype, dtype)
        self.assertEqual(var.get_value().dtype, dtype)
        self.assertEqual(var.fill_value, fill_value)

        var = Variable('foo', value=[4, 5, 6], dimensions='dim')
        self.assertEqual(var.shape, (3,))
        self.assertEqual(var.dtype, var.get_value().dtype)
        self.assertEqual(var.fill_value, var.fill_value)
        sub = var[1]
        self.assertEqual(sub.shape, (1,))

        # Test mask is shared.
        value = [1, 2, 3]
        value = np.ma.array(value, mask=[False, True, False], dtype=float)
        var = Variable(value=value, dtype=int, name='the_name', dimensions='three')
        self.assertNumpyAll(var.get_mask(), value.mask)
        self.assertEqual(var.get_value().dtype, int)

        # Test with bounds.
        desired_bounds_value = [[0.5, 1.5], [1.5, 2.5]]
        bounds = Variable(value=desired_bounds_value, name='n_bnds', dimensions=['ens_dims', 'bounds'])
        var = Variable(value=[1, 2], bounds=bounds, name='n', dimensions='ens_dims')
        self.assertNumpyAll(var.bounds.get_value(), np.array(desired_bounds_value))
        self.assertEqual(list(var.parent.keys()), ['n', 'n_bnds'])
        self.assertEqual(var.attrs['bounds'], bounds.name)

        # Test with a unique identifier.
        var = Variable('woot', uid=50)
        self.assertEqual(var.uid, 50)

        # Test setting is_empty to True.
        v = Variable(is_empty=True)
        self.assertTrue(v.is_empty)

    def test_init_dimensions_named_the_same(self):
        dim = Dimension('same', 7)
        with self.assertRaises(ValueError):
            Variable(name='samed', dimensions=(dim, dim))

    def test_init_dtype(self):
        """Test automatic data type detection."""

        var = Variable(name='hello')
        self.assertIsNone(var.dtype)

        desired = np.int8
        var = Variable(name='foo', value=np.array([1, 2, 3], dtype=desired), dimensions='one')
        self.assertEqual(var.dtype, desired)
        self.assertEqual(var._dtype, 'auto')

    def test_init_fill_value(self):
        """Test automatic fill value creation."""

        var = Variable(name='hello')
        self.assertIsNone(var.fill_value)

        var = Variable(name='hello', fill_value=12)
        self.assertEqual(var.fill_value, 12)

        var = Variable(name='a', value=[1, 2, 3], dimensions='one')
        self.assertEqual(var.fill_value, 999999)

        var = Variable(name='b', value=np.array([1, 2, 3], dtype=object), dimensions='foo')
        self.assertIsNone(var.fill_value)

    def test_init_hang_with_dimension(self):
        # Test passing dimension to value.
        with self.assertRaises(ValueError):
            _ = Variable(name='data', value=Dimension('foo', 5), dimensions='blah')

    def test_init_object_array(self):
        value = [[1, 3, 5],
                 [7, 9],
                 [11]]
        v = Variable(value=value, fill_value=4, name='objects', dimensions='three')
        self.assertEqual(v.dtype, ObjectType(object))
        self.assertEqual(v.shape, (3,))
        for idx in range(v.shape[0]):
            actual = v[idx].get_value()[0]
            desired = value[idx]
            self.assertEqual(actual, desired)

        # Test converting object arrays.
        v = Variable(value=value, dtype=ObjectType(float), name='convert', dimensions='three')
        self.assertEqual(v.get_value()[0].dtype, ObjectType(float))

        v = Variable(value=value, name='foo', dtype=ObjectType(np.float32), dimensions='three')
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            v.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            desired = ds.variables['foo'][:]
        for idx in np.arange(v.shape[0]):
            self.assertNumpyAll(np.array(v.get_value()[idx]), desired[idx])
        v_actual = SourcedVariable(request_dataset=RequestDataset(uri=path, variable='foo'), name='foo')

        actual = v[1].get_masked_value()[0]
        desired = np.array(value[1], dtype=np.float32)
        self.assertNumpyAll(desired, actual)

        for idx in range(v.shape[0]):
            a = v_actual[idx].get_value()[0]
            d = v[idx].get_value()[0]
            self.assertNumpyAll(a, d)

    def test_system_empty_dimensions(self):
        d1 = Dimension('three', 3)
        d2 = Dimension('is_empty', 0, dist=True)
        d2.convert_to_empty()
        d3 = Dimension('what', 10)

        var = Variable('tester', dimensions=[d1, d2, d3])
        self.assertTrue(var.is_empty)
        self.assertEqual(var.shape, (3, 0, 10))

        self.assertIsNone(var.get_value())
        self.assertIsNone(var._value)
        self.assertIsNone(var.get_mask())
        self.assertIsNone(var._mask)

    @attr('data')
    def test_system_remove_variable_from_field(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        time_bounds_name = field.time.bounds.name
        field.remove_variable(field.time)
        self.assertIsNone(field.time)
        self.assertNotIn(time_bounds_name, field)

    @attr('mpi', 'mpi-2')
    def test_system_scatter_gather_variable(self):
        """Test a proof-of-concept scatter and gather operation on a simple variable."""

        if MPI_SIZE != 2:
            raise SkipTest('mpi-2 only')

        slices = [slice(0, 3), slice(3, 5)]
        var_value = np.arange(5)

        if MPI_RANK == 0:
            dim = Dimension('five', 5)
            var = Variable('five', value=var_value, dimensions=dim)
            to_scatter = [var[slc] for slc in slices]
        else:
            to_scatter = None

        lvar = MPI_COMM.scatter(to_scatter)

        self.assertNumpyAll(lvar.get_value(), var_value[slices[MPI_RANK]])
        self.assertEqual(lvar.dimensions[0], Dimension('five', 5)[slices[MPI_RANK]])

        gathered = MPI_COMM.gather(lvar)

        if MPI_RANK == 0:
            size = sum([len(v.dimensions[0]) for v in gathered])
            new_dim = Dimension(gathered[0].dimensions[0].name, size)
            new_value = hgather([v.get_value() for v in gathered])
            new_var = Variable(gathered[0].name, value=new_value, dimensions=new_dim)
            self.assertNumpyAll(new_var.get_value(), var.get_value())
            self.assertEqual(new_var.dimensions[0], dim)

    def test_system_string_data(self):
        var = Variable(name='strings', value=['a.nc', 'b.nc'], dtype=str, dimensions='dstr')
        path = self.get_temporary_file_path('foo.nc')
        var.parent.write(path)
        invar = RequestDataset(path).get()['strings']
        self.assertEqual(invar.dtype, 'S1')

        value = [['2', '0', '1', '3', '-', '0', '3', '-', '0', '1', 'T', '0', '0', ':', '0', '0', ':', '0', '0'],
         ['2', '0', '1', '3', '-', '0', '3', '-', '0', '1', 'T', '0', '0', ':', '3', '0', ':', '0', '0']]
        var = Variable(name='foo', value=value, dtype='S1', dimensions=['time', 'date_len'])
        var.set_string_max_length_global()
        self.assertIsNotNone(var.string_max_length_global)
        self.assertEqual(var.string_max_length_global, 19)

    @attr('mpi')
    def test_system_string_data_parallel(self):
        if vm.size > 2:
            raise SkipTest('vm.size > 2')

        dist = OcgDist()
        sdim = dist.create_dimension('sdim', 2, dist=True)
        dist.update_dimension_bounds(min_elements=1)

        if vm.rank == 0:
            var = Variable(name='strings', value=['peas', 'peas please'], dimensions=sdim.name)
            desired = len(var.get_value()[1])
        else:
            var = None
            desired = None
        desired = vm.bcast(desired)
        var = variable_scatter(var, dist)
        self.assertTrue(var.is_string_object)

        self.assertIsNone(var.string_max_length_global)
        var.set_string_max_length_global()
        self.assertEqual(var.string_max_length_global, desired)

    @attr('mpi')
    def test_system_with_distributed_dimensions_from_file_netcdf(self):
        """Test a distributed read from file."""

        with vm.scoped('desired data write', [0]):
            if not vm.is_null:
                path = self.get_temporary_file_path('dist.nc')
                value = np.arange(5 * 3) * 10 + 1
                desired_sum = value.sum()
                value = value.reshape(5, 3)
                var = Variable('has_dist_dim', value=value, dimensions=['major', 'minor'])
                var.write(path)
            else:
                path = None

        path = MPI_COMM.bcast(path, root=0)
        self.assertTrue(os.path.exists(path))

        rd = RequestDataset(uri=path)

        ompi = OcgDist()
        major = ompi.create_dimension('major', size=5, dist=True, src_idx='auto')
        minor = ompi.create_dimension('minor', size=3, dist=False, src_idx='auto')
        fvar = SourcedVariable(name='has_dist_dim', request_dataset=rd, dimensions=[major, minor])
        ompi.update_dimension_bounds()

        self.assertTrue(fvar.dimensions[0].dist)
        self.assertFalse(fvar.dimensions[1].dist)
        if MPI_RANK <= 1:
            self.assertFalse(fvar.is_empty)
            self.assertIsNotNone(fvar.get_value())
            self.assertEqual(fvar.shape[1], 3)
            if MPI_SIZE == 2:
                self.assertLessEqual(fvar.shape[0], 3)
        else:
            self.assertTrue(fvar.is_empty)

        values = MPI_COMM.gather(fvar.get_value(), root=0)
        if MPI_RANK == 0:
            values = [v for v in values if v is not None]
            values = [v.flatten() for v in values]
            values = hgather(values)
            self.assertEqual(values.sum(), desired_sum)
        else:
            self.assertIsNone(values)

        MPI_COMM.Barrier()

    @attr('mpi')
    def test_system_with_distributed_dimensions_ndvariable(self):
        """Test multi-dimensional variable behavior with distributed dimensions."""

        d1 = Dimension('d1', size=5, dist=True)
        d2 = Dimension('d2', size=10, dist=False)
        d3 = Dimension('d3', size=3, dist=True)
        dimensions = [d1, d2, d3]
        ompi = OcgDist()
        for d in dimensions:
            ompi.add_dimension(d)
        ompi.update_dimension_bounds()

        var = Variable('ndist', dimensions=dimensions)

        if MPI_RANK > 1:
            self.assertTrue(var.is_empty)
        else:
            self.assertFalse(var.is_empty)

        MPI_COMM.Barrier()

    def test_system_parents_on_bounds_variable(self):
        extra = self.get_variable(return_original_data=False)
        parent = VariableCollection(variables=extra)
        bounds = Variable(value=[[1, 2], [3, 4]], name='the_bounds', parent=parent, dimensions=['a', 'b'])

        extra2 = Variable(value=7.0, name='remember', dimensions=[])
        parent = VariableCollection(variables=extra2)
        var = Variable(name='host', value=[1.5, 3.5], dimensions='a', bounds=bounds, parent=parent)
        self.assertEqual(list(var.parent.keys()), ['remember', 'host', 'time_value', 'the_bounds'])

    def test_system_subsetting_single_value(self):
        # Test using data loaded from file.
        field = self.get_field(nlevel=1)
        level_value = field.level.get_value()[0]
        path = self.get_temporary_file_path('foo.nc')
        field.write(path)
        infield = RequestDataset(path).get()
        infield.level.get_between(level_value - 1, level_value + 1)
        self.assertEqual(infield.level.get_value()[0], level_value)

        # Test through operations.
        level_range = [int(level_value - 1), int(level_value + 1)]
        ops = OcgOperations(dataset={'uri': path}, level_range=level_range, output_format='nc')
        actual = ops.execute()
        actual = RequestDataset(actual).get().level.get_value()[0]
        self.assertEqual(actual, level_value)

    def test_allocate_value(self):
        dim = Dimension('foo', 5)
        var = Variable(name='tester', dimensions=dim, dtype=np.int8)
        var.allocate_value()
        actual = var.get_value()
        self.assertTrue(np.all(actual == var.fill_value))

    def test_as_record(self):
        var = Variable(name='foo', value=[1], dimensions='one',
                       repeat_record=OrderedDict([('key', 'min'), ('src', 'foo_origin')]))

        desired = OrderedDict([('DID', None), ('foo', 1), ('key', 'min'), ('src', 'foo_origin')])
        actual = var._as_record_()
        self.assertDictEqual(actual, desired)

    def test_bounds(self):
        # Test adding/removing bounds.
        var = Variable('bounded', value=[5], dimensions='one')
        self.assertNotIn('bounds', var.attrs)
        var.set_bounds(Variable('bds', value=[[6, 7]], dimensions=['one', 'bounds']))
        self.assertEqual(var.attrs['bounds'], 'bds')
        var.set_bounds(None)
        self.assertNotIn('bounds', var.attrs)

    @attr('cfunits')
    def test_cfunits(self):
        var = self.get_variable(return_original_data=False)
        actual = get_units_object(var.units)
        self.assertTrue(get_are_units_equal((var.cfunits, actual)))

    @attr('cfunits')
    def test_cfunits_conform(self):
        units_kelvin = get_units_object('kelvin')
        original_value = np.array([5, 5, 5])

        # Conversion of celsius units to kelvin.
        var = Variable(name='tas', units='celsius', value=original_value, dimensions='three')
        self.assertEqual(len(var.attrs), 1)
        var.cfunits_conform(units_kelvin)
        self.assertNumpyAll(var.get_masked_value(), np.ma.array([278.15] * 3, fill_value=var.fill_value))
        self.assertEqual(var.cfunits, units_kelvin)
        self.assertEqual(var.units, 'kelvin')
        self.assertEqual(len(var.attrs), 1)

        # If there are no units associated with a variable, conforming the units should fail.
        var = Variable(name='tas', units=None, value=original_value, dimensions='three')
        with self.assertRaises(NoUnitsError):
            var.cfunits_conform(units_kelvin)

        # Conversion should fail for nonequivalent units.
        var = Variable(name='tas', units='kelvin', value=original_value, dimensions='three')
        with self.assertRaises(ValueError):
            var.cfunits_conform(get_units_object('grams'))

        # The data type should always be updated to match the output from CF units backend.
        av = Variable(value=np.array([4, 5, 6]), dtype=int, name='what', dimensions='three')
        self.assertEqual(av.dtype, np.dtype(int))
        with self.assertRaises(NoUnitsError):
            av.cfunits_conform('K')
        av.units = 'celsius'
        av.cfunits_conform('K')
        self.assertEqual(av._dtype, 'auto')
        self.assertEqual(av.dtype, av.get_value().dtype)

        # Test with bounds.
        bv = Variable(value=[5., 10., 15.], units='celsius', name='tas', dimensions=['ll'])
        bv.set_extrapolated_bounds('the_bounds', 'bounds')
        self.assertEqual(bv.bounds.units, 'celsius')
        bv.cfunits_conform(get_units_object('kelvin'))
        self.assertEqual(bv.bounds.units, 'kelvin')
        self.assertNumpyAll(bv.bounds.get_masked_value(),
                            np.ma.array([[275.65, 280.65], [280.65, 285.65], [285.65, 290.65]]))

        # Test conforming without bounds.
        bv = Variable(value=[5., 10., 15.], units='celsius', name='tas', dimensions='three')
        bv.cfunits_conform('kelvin')
        self.assertNumpyAll(bv.get_masked_value(), np.ma.array([278.15, 283.15, 288.15]))

    @attr('cfunits')
    def test_cfunits_conform_masked_array(self):
        # Assert mask is respected by unit conversion.
        value = np.ma.array(data=[5, 5, 5], mask=[False, True, False])
        var = Variable(name='tas', units=get_units_object('celsius'), value=value, dimensions='three')
        var.cfunits_conform(get_units_object('kelvin'))
        desired = np.ma.array([278.15, 278.15, 278.15], mask=[False, True, False], fill_value=var.fill_value)
        self.assertNumpyAll(var.get_masked_value(), desired)

    def test_convert_to_empty(self):
        var = Variable(value=[1, 2, 3], mask=[True, False, True], dimensions='alpha', name='to_convert')
        self.assertEqual(var.ndim, 1)
        var.convert_to_empty()
        self.assertTrue(var.is_empty)
        self.assertIsNone(var.get_value())
        self.assertIsNone(var.get_mask())
        self.assertIsNotNone(var.dimensions)
        self.assertEqual(var.ndim, 1)

        var.dimensions[0].dist = True
        var.dimensions[0].convert_to_empty()
        self.assertEqual(var.shape, (0,))
        self.assertEqual(var.ndim, 1)

    def test_copy(self):
        var = Variable(value=[5], mask=[True], name='copycat', dimensions='uni')
        cvar = var.copy()
        self.assertNotEqual(id(var), id(cvar))
        self.assertNotEqual(id(var.parent), id(cvar.parent))
        self.assertNotEqual(id(var.get_value()), id(cvar.get_value()))
        self.assertNotEqual(id(var.get_mask()), id(cvar.get_mask()))
        cvar.set_value([10])
        self.assertNotEqual(var.get_value()[0], cvar.get_value()[0])
        cvar.attrs['new_attr'] = 'new'
        self.assertNotIn('new_attr', var.attrs)
        cvar.set_dimensions(Dimension('overload', 1))
        self.assertNotIn('overload', var.parent.dimensions)

        # Test this is not a deepcopy.
        var = Variable(value=[10, 11, 12], dimensions=Dimension('three', 3, src_idx=np.arange(3)),
                       name='no deepcopies!')
        cvar = var.copy()
        self.assertNumpyMayShareMemory(var.get_value(), cvar.get_value())
        self.assertNumpyMayShareMemory(var.dimensions[0]._src_idx, cvar.dimensions[0]._src_idx)

        # Test copied parent does not share reference.
        var = Variable(name='foo', value=[1, 2], dimensions='two')
        var2 = Variable(name='foo1', value=[5, 6, 7], dimensions='three')
        var_copy = var.copy()
        var_copy.parent.add_variable(var2)
        self.assertNotIn(var2.name, var.parent)

        # Test dimensions are not shared.
        var = Variable(name='foo', value=[1, 2], dimensions='two')
        var_copy = var.copy()
        var_copy.parent.add_dimension(Dimension('dummy'))
        self.assertNotIn('dummy', var.parent.dimensions)

    def test_deepcopy(self):
        var = Variable(value=[5], dimensions='one', name='o')
        var.set_bounds(Variable(value=[[4, 6]], dimensions=['one', 'bnds'], name='bounds'))
        misc = Variable(value=np.arange(8), dimensions='eight', name='dont_touch_me')
        var.parent.add_variable(misc)

        dvar = var.deepcopy()
        self.assertFalse(np.may_share_memory(var.get_value(), dvar.get_value()))
        self.assertFalse(np.may_share_memory(var.bounds.get_value(), dvar.bounds.get_value()))

    def test_dimensions(self):
        # Test dimensions on bounds variables are updated when the dimensions on the parent variable are updated.
        aa = Dimension('aa', 4, src_idx=np.array([11, 12, 13, 14]))
        var = Variable('bounded', value=[1, 2, 3, 4], dimensions=aa)
        var.set_extrapolated_bounds('aa_bounds', 'aabnds')
        var.set_dimensions(Dimension('bb', 4))
        self.assertEqual(var.dimensions[0], var.bounds.dimensions[0])

    def test_extent(self):
        var = Variable(name='a', value=[1, 2, 3, 4, 5], dimensions='b')
        self.assertEqual(var.extent, (1, 5))

    def test_extract(self):
        ancillary = Variable('ancillary')
        src = Variable('src', parent=ancillary.parent)
        csrc = src.copy()
        csrc = csrc.extract()

        self.assertNotIn(ancillary.name, csrc.parent)
        self.assertIn(ancillary.name, src.parent)

        # Test with a clean break.
        v1 = Variable('one')
        v2 = Variable('two')
        v3 = Variable('three')
        vc = VariableCollection(variables=[v1, v2, v3])
        v1e = v1.extract(clean_break=True)
        self.assertNotIn(v1.name, vc)
        self.assertEqual(list(vc.keys()), ['two', 'three'])

        # Test dimensions match the extracted variable only.
        var1 = Variable('var1', value=[1, 2], dimensions='dim1')
        var2 = Variable('var2', value=[1, 2, 3], dimensions='dim2')
        _ = VariableCollection(variables=[var1, var2])
        var_extract = var1.extract()
        self.assertEqual(len(var_extract.parent.dimensions), 1)
        self.assertEqual(var_extract.parent.dimensions[var1.dimensions[0].name].name, var1.dimensions[0].name)

    def test_get_between(self):
        bv = Variable('foo', value=[0], dimensions='uni')
        with self.assertRaises(EmptySubsetError):
            bv.get_between(100, 200)

        dim = Dimension('a', 4, src_idx='auto')
        bv = Variable('foo', value=[100, 200, 300, 400], dimensions=dim)
        vdim_between = bv.get_between(100, 200)
        self.assertEqual(vdim_between.shape[0], 2)
        self.assertNumpyMayShareMemory(bv.get_value(), vdim_between.get_value())
        actual = vdim_between.dimensions[0]._src_idx
        self.assertEqual(actual, (0, 2))

    def test_get_between_bounds(self):
        value = [0., 5., 10.]
        bounds = [[-2.5, 2.5], [2.5, 7.5], [7.5, 12.5]]

        # A reversed copy of these bounds are created here.
        value_reverse = deepcopy(value)
        value_reverse.reverse()
        bounds_reverse = deepcopy(bounds)
        bounds_reverse.reverse()
        for ii in range(len(bounds)):
            bounds_reverse[ii].reverse()

        data = {'original': {'value': value, 'bounds': bounds},
                'reversed': {'value': value_reverse, 'bounds': bounds_reverse}}
        for key in ['original', 'reversed']:
            bounds = Variable('hello_bounds', value=data[key]['bounds'], dimensions=['a', 'b'])
            vdim = Variable('hello', value=data[key]['value'], bounds=bounds, dimensions=['a'])

            vdim_between = vdim.get_between(1, 3)
            self.assertEqual(vdim_between.size, 2)

            actual = vdim_between.bounds.get_value()

            if key == 'original':
                desired = [[-2.5, 2.5], [2.5, 7.5]]
                self.assertEqual(actual.tolist(), desired)
            else:
                self.assertEqual(actual.tolist(), [[7.5, 2.5], [2.5, -2.5]])
            self.assertEqual(vdim.resolution, 5.0)

            # Preference is given to the lower bound in the case of "ties" where the value could be assumed part of the
            # lower or upper cell.
            vdim_between = vdim.get_between(2.5, 2.5)
            self.assertEqual(vdim_between.size, 1)
            actual = vdim_between.bounds.get_masked_value()
            if key == 'original':
                self.assertNumpyAll(actual, np.ma.array([[2.5, 7.5]]))
            else:
                self.assertNumpyAll(actual, np.ma.array([[7.5, 2.5]]))

            # If the interval is closed and the subset range falls only on bounds value then the subset will be empty.
            with self.assertRaises(EmptySubsetError):
                vdim.get_between(2.5, 2.5, closed=True)

            vdim_between = vdim.get_between(2.5, 7.5)
            actual = vdim_between.bounds.get_value()
            if key == 'original':
                self.assertEqual(actual.tolist(), [[2.5, 7.5], [7.5, 12.5]])
            else:
                self.assertEqual(actual.tolist(), [[12.5, 7.5], [7.5, 2.5]])

    def test_get_between_bounds_again(self):
        """Test for simpler bounds subsettting."""

        var = Variable(name='data', value=[1, 2, 3, 4], dtype=float, dimensions='dim')
        var.set_extrapolated_bounds('data_bounds', 'bounds')
        sub = var.get_between(2., 3.)

    def test_get_between_use_bounds(self):
        value = [3., 5.]
        bounds = [[2., 4.], [4., 6.]]
        bounds = Variable('bounds', bounds, dimensions=['a', 'b'])
        vdim = Variable('foo', value=value, bounds=bounds, dimensions=['a'])
        ret = vdim.get_between(3, 4.5, use_bounds=False)
        self.assertNumpyAll(ret.get_masked_value(), np.ma.array([3.]))
        self.assertNumpyAll(ret.bounds.get_masked_value(), np.ma.array([[2., 4.]]))

    def test_get_between_single_value(self):
        var = Variable(name='tester', value=[3], dimensions='level')
        sub = var.get_between(1, 5)
        self.assertEqual(var.shape, sub.shape)
        self.assertNumpyAll(var.get_value(), sub.get_value())

    @attr('mpi')
    def test_get_distributed_slice(self):
        if MPI_RANK == 0:
            path = self.get_temporary_file_path('out.nc')
        else:
            path = None
        path = MPI_COMM.bcast(path)

        dist = OcgDist()
        dim1 = dist.create_dimension('time', 10, dist=False, src_idx='auto')
        dim2 = dist.create_dimension('x', 5, dist=True, src_idx='auto')
        dim3 = dist.create_dimension('y', 4, dist=False, src_idx='auto')
        var = dist.create_variable('var', dimensions=[dim1, dim2, dim3])
        dist.update_dimension_bounds()

        if not var.is_empty:
            var.get_value()[:] = MPI_RANK

        with vm.scoped_by_emptyable('var slice', var):
            if not vm.is_null:
                sub = var.get_distributed_slice([slice(None), slice(2, 4), slice(3, 4)])
                vc = VariableCollection(variables=[sub])
                vc.write(path)

                rd = RequestDataset(path)
                svar = SourcedVariable('var', request_dataset=rd)
                self.assertEqual(svar.shape, (10, 2, 1))

    @attr('mpi')
    def test_get_distributed_slice_fancy_indexing(self):
        """Test using a fancy indexing slice in parallel."""

        if vm.size != 2:
            raise SkipTest('vm.size != 2')

        dist = OcgDist()
        dim1 = dist.create_dimension('dim1', size=7, dist=True, src_idx='auto')
        dim2 = dist.create_dimension('dim2', size=4)
        dist.update_dimension_bounds()

        if vm.rank == 0:
            value = np.zeros((7, 4))
            for idx in range(value.shape[0]):
                value[idx, :] = idx
            var = Variable(name='test', dimensions=['dim1', 'dim2'], value=value)
        else:
            var = None
        var = variable_scatter(var, dist)

        slices = {0: np.array([False, True, False, True]),
                  1: np.array([False, True, False])}
        for k, v in slices.items():
            v = [v, slice(None)]
            slices[k] = v

        sub = var.get_distributed_slice(slices[vm.rank])

        gvar = variable_gather(sub)

        if vm.rank == 0:
            desired = [[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0], [5.0, 5.0, 5.0, 5.0]]
            actual = gvar.get_value().tolist()
            self.assertEqual(actual, desired)

        # Test when all are False.
        slices = {0: np.array([False, False, False, False]),
                  1: np.array([True, True, True])}
        for k, v in slices.items():
            v = [v, slice(None)]
            slices[k] = v
        sub = var.get_distributed_slice(slices[vm.rank])
        if vm.rank == 0:
            self.assertTrue(sub.is_empty)
        else:
            self.assertFalse(sub.is_empty)
            self.assertEqual(sub.get_value().sum(), 60)
            self.assertEqual(sub.dimensions[0]._src_idx.tolist(), [4, 5, 6])

    @attr('mpi')
    def test_get_distributed_slice_parent_variables(self):
        """Test distributed slicing on a variable with sibling variables."""
        dist = OcgDist()
        dim = dist.create_dimension('dim', 9, dist=True)
        dim_nondist = dist.create_dimension('dim_nondist', 3, dist=False)
        dist.update_dimension_bounds()

        live_ranks = vm.get_live_ranks_from_object(dim)
        vm.create_subcomm('parent slicing', live_ranks, is_current=True)
        if vm.is_null:
            self.assertTrue(dim.is_empty)
            return

        value1 = np.arange(*dim.bounds_local)
        value2 = value1 + 100
        value4 = np.zeros((dim_nondist.size, dim.bounds_local[1] - dim.bounds_local[0]), dtype=int)
        for ii in range(dim_nondist.size):
            value4[ii, :] = value1

        var1 = Variable(name='var1', value=value1, dimensions=dim)
        var2 = Variable(name='var2', value=value2, dimensions=dim)
        var3 = Variable(name='var3', value=[230, 231, 232], dimensions=dim_nondist)
        var4 = Variable(name='var4', value=value4, dimensions=(dim_nondist, dim))

        VariableCollection(variables=[var1, var2, var3, var4])

        for _ in range(10):
            sub = var1.get_distributed_slice(7)

            non_empty = {1: 0, 2: 1, 3: 2, 5: 3}
            desired = {'var1': [7], 'var2': [107], 'var3': var3.get_value().tolist(), 'var4': [[7], [7], [7]]}
            if MPI_RANK == non_empty.get(MPI_SIZE, 3):
                actual = {var.name: var.get_value().tolist() for var in list(sub.parent.values())}
                self.assertEqual(actual, desired)
            else:
                self.assertTrue(sub.is_empty)

    @attr('mpi')
    def test_get_distributed_slice_simple(self):
        self.add_barrier = False
        if MPI_RANK == 0:
            path = self.get_temporary_file_path('out.nc')
        else:
            path = None
        path = MPI_COMM.bcast(path)

        dist = OcgDist()
        dim = dist.create_dimension('x', 5, dist=True, src_idx='auto')
        var = dist.create_variable('var', dimensions=[dim])
        dist.update_dimension_bounds(min_elements=1)

        if not var.is_empty:
            var.get_value()[:] = (MPI_RANK + 1) ** 3 * (np.arange(var.shape[0]) + 1)

        live_ranks = vm.get_live_ranks_from_object(var)
        vm.create_subcomm('simple slice test', live_ranks, is_current=True)

        if not vm.is_null:
            sub = var.get_distributed_slice(slice(2, 4))
        else:
            sub = Variable(is_empty=True)

        if MPI_SIZE == 3:
            if MPI_RANK != 1:
                self.assertTrue(sub.is_empty)
            else:
                self.assertFalse(sub.is_empty)

        if not vm.is_null:
            live_ranks = vm.get_live_ranks_from_object(sub)
            vm.create_subcomm('sub write', live_ranks, is_current=True)
            if not vm.is_null:
                vc = VariableCollection(variables=[sub])
                vc.write(path)

        MPI_COMM.Barrier()

        vm.finalize()
        vm.__init__()

        with vm.scoped('read data from file', [0]):
            if not vm.is_null and vm.rank == 0:
                rd = RequestDataset(path)
                svar = SourcedVariable('var', request_dataset=rd)
                self.assertEqual(svar.shape, (2,))
                if MPI_SIZE == 1:
                    self.assertEqual(svar.get_value().tolist(), [3., 4.])
                elif MPI_SIZE == 2:
                    self.assertEqual(svar.get_value().tolist(), [3., 8.])
                elif MPI_SIZE == 3:
                    self.assertEqual(svar.get_value().tolist(), [8., 16.])
                elif MPI_SIZE == 8:
                    self.assertEqual(svar.get_value().tolist(), [27., 64.])

    def test_get_iter(self):
        # Test two repeat records.
        var = Variable(name='a', value=[1, 2], repeat_record=[('UGID', 40)], dimensions='two')
        itr = var.get_iter(repeaters=[('hola', 'amigo')])
        for record in itr:
            self.assertIn('hola', record)

    def test_get_mask(self):
        var = Variable(value=[1, 2, 3], mask=[False, True, False], name='masked', dimensions='three')
        assert_equal(var.get_mask(), [False, True, False])
        value = np.ma.array([1, 2, 3], mask=[False, True, False])
        var = Variable(value=value, name='v', dimensions='v')
        assert_equal(var.get_mask(), [False, True, False])

        var = Variable(value=[1, 2, 3], name='three', dimensions='three')
        self.assertIsNone(var.get_mask())
        cpy = var.copy()
        cpy.set_mask([False, True, False])
        cpy.get_value().fill(10)
        self.assertTrue(np.all(var.get_value() == 10))
        self.assertIsNone(var.get_mask())

        var = Variable(value=np.random.rand(2, 3, 4), fill_value=200, name='random',
                       dimensions=['two', 'three', 'four'])
        var.get_value()[1, 1] = 200
        self.assertTrue(np.all(var.get_mask(create=True, check_value=True)[1, 1]))
        self.assertEqual(var.get_mask().sum(), 4)

        # Test with bounds.
        bv = self.get_boundedvariable()
        bv.set_mask([False, True, False])
        bounds_mask = bv.bounds.get_mask()
        self.assertTrue(np.all(bounds_mask[1, :]))
        self.assertEqual(bounds_mask.sum(), 2)

        # Test with two dimensions.
        bv = self.get_boundedvariable_2d()
        self.assertEqual(bv.bounds.ndim, 3)
        mask = np.array([[False, True],
                         [False, False],
                         [True, False]], dtype=bool)
        bv.set_mask(mask)
        bounds_mask = bv.bounds.get_mask()
        for slc in ((0, 1), (2, 0)):
            self.assertTrue(np.all(bounds_mask[slc]))
        self.assertEqual(bounds_mask.sum(), 8)

        # Test reference is shared.
        vars = [Variable('foo', [1, 2, 3], 'three', float, [False, False, False]),
                Variable('foo', np.ma.array([1, 2, 3], mask=[False, False, False]), 'three', float),
                Variable('foo', np.ma.array([1, 2, 3], mask=False), 'three', float)]
        for var in vars:
            vmask = var.get_mask()
            vmask[1] = True
            self.assertEqual(id(vmask), id(var.get_mask()))
            self.assertTrue(var.get_mask()[1])
            self.assertTrue(var._mask[1])

    def test_getitem(self):
        var = Variable(value=[1, 2, 3], name='three', dimensions='three')
        sub = var[1]
        self.assertEqual(var.shape, (3,))
        self.assertEqual(sub.shape, (1,))

        # Test a dictionary slice.
        var = Variable(name='empty')
        dslc = {'one': slice(2, 5), 'two': np.array([False, True, False, True]), 'three': 0}
        with self.assertRaises(DimensionsRequiredError):
            var[dslc]
        value = np.ma.arange(5 * 4 * 7 * 10, fill_value=100).reshape(5, 4, 10, 7)
        var = Variable(value=value, name='range', dimensions=['one', 'two', 'four', 'three'])
        sub = var[dslc]
        self.assertEqual(sub.shape, (3, 2, 10, 1))
        sub_value = value[2:5, np.array([False, True, False, True], dtype=bool), slice(None), slice(0, 1)]
        self.assertNumpyAll(sub.get_masked_value(), sub_value)

        # Test with a parent.
        var = Variable(name='a', value=[1, 2, 3], dimensions=['one'])
        parent = VariableCollection(variables=[var])
        var2 = Variable(name='b', value=[11, 22, 33], dimensions=['one'], parent=parent)
        self.assertIn('b', var2.parent)
        sub = var2[1]
        self.assertEqual(var2.parent.shapes, OrderedDict([('a', (3,)), ('b', (3,))]))
        self.assertEqual(sub.parent.shapes, OrderedDict([('a', (1,)), ('b', (1,))]))

        # Test slicing a bounded variable.
        bv = self.get_boundedvariable()
        sub = bv[1]
        self.assertEqual(sub.bounds.shape, (1, 2))
        self.assertNumpyAll(sub.bounds.get_value(), bv.bounds[1, :].get_value())

        # Test with a boolean array.
        var = Variable(value=[1, 2, 3, 4, 5], name='five', dimensions='five')
        sub = var[[False, True, True, True, False]]
        self.assertEqual(sub.shape, (3,))

        # Test filling the value on a sliced variable.
        var = Variable(name='foo', dimensions=Dimension('one', 3), dtype=object)
        self.assertIsNone(var._value)
        var.allocate_value()
        sub = var[{'one': 1}]
        sub.get_value()[:] = 100.
        self.assertEqual(var.get_value()[1], 100.)

        # Test dimension bounds are updated.
        var = Variable(name='foo', value=np.zeros((31, 360, 720)), dimensions=['time', 'row', 'col'])
        desired_bounds_global = [(0, 31), (0, 360), (0, 720)]
        bounds_global = [d.bounds_global for d in var.dimensions]
        self.assertEqual(bounds_global, desired_bounds_global)
        slc = {'time': 1, 'row': slice(0, 180), 'col': slice(0, 360)}
        desired_bounds_global_slice = [(0, 1), (0, 180), (0, 360)]
        sub = var[slc]
        bounds_global = [d.bounds_global for d in sub.dimensions]
        self.assertEqual(desired_bounds_global_slice, bounds_global)
        self.assertEqual(desired_bounds_global_slice, [d.bounds_local for d in sub.dimensions])

    def test_getitem_index_slice(self):
        # Test with an index slice.
        d_coords = Dimension('d_coords', 7, src_idx='auto')
        coords = Variable(name='coords', value=[0, 11, 22, 33, 44, 55, 66], dimensions=d_coords)
        d_cindex = Dimension('d_cindex', 9, src_idx='auto')
        index = Variable(name='cindex', value=[4, 2, 1, 2, 1, 4, 1, 4, 2], dimensions=d_cindex, parent=coords.parent)

        sub_coords = coords[index.get_value()]
        self.assertEqual(sub_coords.dimensions[0].size, 9)

        actual = sub_coords.get_value().tolist()
        desired = [44, 22, 11, 22, 11, 44, 11, 44, 22]
        self.assertEqual(actual, desired)

        actual = sub_coords.dimensions[0]._src_idx.tolist()
        desired = index.get_value().tolist()
        self.assertEqual(actual, desired)

    def test_group(self):
        var = Variable(name='lonely')
        self.assertEqual(var.group, [None])

        vc1 = VariableCollection(name='one')
        vc2 = VariableCollection(name='two', parent=vc1)
        vc3 = VariableCollection(name='three', parent=vc2)
        var = Variable(name='lonely')
        vc3.add_variable(var)
        self.assertEqual(var.group, ['one', 'two', 'three'])

    def test_set_bounds(self):

        for l in ['__default__', False]:
            if l == '__default__':
                pass
            else:
                env.CLOBBER_UNITS_ON_BOUNDS = l

            var = Variable(name='hi', value=[2], dimensions='one', units='unique')
            bnds = Variable(name='hi_bounds', value=[[1, 3]], dimensions=['one', 'bounds'])

            var.set_bounds(bnds)

            if l == '__default__':
                self.assertEqual(var.units, bnds.units)
            else:
                self.assertIsNone(bnds.units)

    def test_set_extrapolated_bounds(self):
        bv = self.get_boundedvariable(mask=[False, True, False])
        self.assertIsNotNone(bv.bounds)
        bv.set_bounds(None)
        self.assertIsNone(bv.bounds)
        bv.set_extrapolated_bounds('x_bounds', 'bounds')
        self.assertIn('x_bounds', bv.parent)
        self.assertEqual(bv.bounds.name, 'x_bounds')
        self.assertEqual(bv.bounds.ndim, 2)
        bounds_mask = bv.bounds.get_mask()
        self.assertTrue(np.all(bounds_mask[1, :]))
        self.assertEqual(bounds_mask.sum(), 2)

        # Test extrapolating bounds on 2d variable.
        bv = self.get_boundedvariable_2d()
        bv.set_bounds(None)
        self.assertIsNone(bv.bounds)
        bv.set_extrapolated_bounds('two_dee_bounds', 'bounds_dimension')
        bounds_value = bv.bounds.get_masked_value()
        actual = [[[2.25, 2.75, 1.75, 1.25], [2.75, 3.25, 2.25, 1.75]],
                  [[1.25, 1.75, 0.75, 0.25], [1.75, 2.25, 1.25, 0.75]],
                  [[0.25, 0.75, -0.25, -0.75], [0.75, 1.25, 0.25, -0.25]]]
        actual = np.ma.array(actual, mask=False)
        self.assertNumpyAll(actual, bounds_value)
        self.assertEqual(bounds_value.ndim, 3)
        bounds_dimensions = bv.bounds.dimensions
        self.assertEqual(bv.bounds.name, 'two_dee_bounds')
        self.assertEqual(len(bounds_dimensions), 3)
        self.assertEqual(bounds_dimensions[2].name, 'bounds_dimension')

    def test_set_value(self):
        # Test setting value to None only allowed for dimensionless.
        var = Variable(name='tester', value=[1, 2, 3], mask=[False, True, False], dimensions=['foo'])
        with self.assertRaises(ValueError):
            var.set_value(None)

        var = Variable(name='tester2', value=4, dimensions=())
        var.set_value(None)
        self.assertIsNone(var.get_value())

    def test_setitem(self):
        var = Variable(value=[10, 10, 10, 10, 10], name='tens', dimensions='dim')
        var2 = Variable(value=[2, 3, 4], mask=[True, True, False], name='var2', dimensions='three')
        var[1:4] = var2
        self.assertEqual(var.get_value().tolist(), [10, 2, 3, 4, 10])
        self.assertEqual(var.get_mask().tolist(), [False, True, True, False, False])

        # Test 2 dimensions.
        value = np.zeros((3, 4), dtype=int)
        var = Variable(value=value, name='two', dimensions=['three', 'four'])
        with self.assertRaises(IndexError):
            var[1] = Variable(value=4500, name='scalar', dimensions=[])
        var[1, 1:3] = Variable(value=6700, name='scalar', dimensions=[])
        self.assertTrue(np.all(var.get_value()[1, 1:3] == 6700))
        self.assertAlmostEqual(var.get_value().mean(), 1116.66666666)

        # Test with bounds.
        bv = self.get_boundedvariable()
        bounds = Variable(value=[[500, 700]], name='b', dimensions=['a', 'b'])
        bv2 = Variable(value=[600], bounds=bounds, name='c', dimensions=['d'])
        bv[1] = bv2
        self.assertEqual(bv.bounds.get_value()[1, :].tolist(), [500, 700])

    def test_shape(self):
        # Test shape with unlimited dimension.
        dim = Dimension('time')
        var = Variable(name='time', value=[4, 5, 6], dimensions=dim)
        self.assertEqual(var.shape, (3,))
        self.assertEqual(len(dim), 3)
        # Copies are made after slicing.
        sub = var[1]
        self.assertEqual(len(dim), 3)
        self.assertEqual(len(sub.dimensions[0]), 1)
        self.assertEqual(sub.shape, (1,))

    def test_units(self):
        var = Variable(name='empty')
        self.assertIsNone(var.units)
        self.assertNotIn('units', var.attrs)
        var.units = 'large'
        self.assertEqual(var.attrs['units'], 'large')
        self.assertEqual(var.units, 'large')
        var.units = 'small'
        self.assertEqual(var.attrs['units'], 'small')
        self.assertEqual(var.units, 'small')
        var.units = None
        self.assertEqual(var.units, None)

        var = Variable(units='haze', name='hazed')
        self.assertEqual(var.units, 'haze')

        var = Variable(name='no units')
        var.units = None
        self.assertEqual(var.attrs['units'], None)

        # Test units behavior with bounds.
        bounds = Variable(value=[[5, 6]], dtype=float, name='bnds', dimensions=['t', 'bnds'], units='celsius')
        var = Variable(name='some', value=[5.5], dimensions='t', bounds=bounds, units='K')
        self.assertEqual(var.bounds.units, 'K')
        var.units = None
        self.assertIsNone(var.bounds.units)

    def test_write_netcdf(self):
        var = self.get_variable(return_original_data=False)
        self.assertIsNotNone(var.fill_value)
        self.assertIsNone(var._mask)
        new_mask = var.get_mask(create=True)
        new_mask[1] = True
        var.set_mask(new_mask)
        self.assertIsNotNone(var.fill_value)
        var.attrs['axis'] = 'not_an_ally'
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            var.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            ncvar = ds.variables[var.name]
            self.assertEqual(ncvar.units, 'kelvin')
            self.assertEqual(ncvar.dtype, var.dtype)
            self.assertTrue(ncvar[:].mask[1])
            self.assertEqual(ncvar.axis, 'not_an_ally')

        # Test writing an unlimited dimension.
        path = self.get_temporary_file_path('foo.nc')
        dim = Dimension('time')
        var = Variable(name='time', value=[4, 5, 6], dimensions=dim)
        self.assertEqual(var.shape, (3,))
        self.assertTrue(var.dimensions[0].is_unlimited)
        for unlimited_to_fixed_size in [False, True]:
            with self.nc_scope(path, 'w') as ds:
                var.write(ds, unlimited_to_fixedsize=unlimited_to_fixed_size)
            # subprocess.check_call(['ncdump', path])
            with self.nc_scope(path) as ds:
                rdim = ds.dimensions['time']
                rvar = ds.variables['time']
                if unlimited_to_fixed_size:
                    self.assertFalse(rdim.isunlimited())
                else:
                    self.assertTrue(rdim.isunlimited())
                # Fill value only present for masked data.
                self.assertNotIn('_FillValue', rvar.__dict__)

        # Test writing with bounds.
        bv = self.get_boundedvariable()
        dim_x = Dimension('x', 3)
        bv.set_dimensions(dim_x)
        bv.bounds.set_dimensions([dim_x, Dimension('bounds', 2)])
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            bv.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path, 'r') as ds:
            var = ds.variables[bv.name]
            self.assertEqual(var.bounds, bv.bounds.name)
            self.assertNumpyAll(ds.variables[bv.bounds.name][:], bv.bounds.get_value())


class TestSourcedVariable(AbstractTestInterface):
    def get_sourcedvariable(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'tas'
        if 'request_dataset' not in kwargs:
            kwargs['request_dataset'] = self.get_request_dataset()
        sv = SourcedVariable(**kwargs)
        self.assertIsNone(sv._value)
        return sv

    @attr('data')
    def test_init(self):
        sv = self.get_sourcedvariable()
        self.assertIsInstance(sv._request_dataset, RequestDataset)
        self.assertEqual(sv.units, 'K')

        sv = self.get_sourcedvariable(name='time_bnds')
        self.assertIsNone(sv._value)
        self.assertEqual(sv.ndim, 2)
        sub = sv[5:10, :]
        self.assertIsNone(sub._value)

        # Test initializing with a value.
        sv = SourcedVariable(value=[1, 2, 3], name='foo', dimensions='three')
        self.assertEqual(sv.dtype, np.int)
        self.assertEqual(sv.get_masked_value().fill_value, 999999)
        self.assertEqual(sv.shape, (3,))
        self.assertEqual(len(sv.dimensions), 1)

        # Test protecting data.
        sv = self.get_sourcedvariable(protected=True)
        with self.assertRaises(PayloadProtectedError):
            sv.get_value()
        self.assertIsNone(sv._value)

    @attr('data')
    def test_init_bounds(self):
        bv = self.get_boundedvariable()
        self.assertEqual(bv.shape, (3,))

        # Test loading from source.
        request_dataset = self.get_request_dataset()
        bounds = SourcedVariable(request_dataset=request_dataset, name='time_bnds', protected=True)
        bv = SourcedVariable(bounds=bounds, name='time', request_dataset=request_dataset, protected=True)
        self.assertEqual(len(bv.dimensions), 1)
        self.assertEqual(len(bv.bounds.dimensions), 2)
        self.assertEqual(bv.bounds.ndim, 2)
        self.assertEqual(bv.ndim, 1)
        bv = bv[30:50]
        self.assertEqual(bv.ndim, 1)
        self.assertEqual(bv.dtype, np.float64)
        self.assertEqual(bv.bounds.dtype, np.float64)
        self.assertEqual(bv.shape, (20,))
        self.assertEqual(bv.bounds.shape, (20, 2))
        self.assertEqual(len(bv.dimensions), 1)
        self.assertEqual(len(bv.bounds.dimensions), 2)
        self.assertIsNone(bv.bounds._value)
        self.assertIsNone(bv._value)

        # Test with two dimensions.
        y_value = [[40.0, 40.0, 40.0], [41.0, 41.0, 41.0], [42.0, 42.0, 42.0], [43.0, 43.0, 43.0]]
        y_corners = [[[39.5, 39.5, 40.5, 40.5], [39.5, 39.5, 40.5, 40.5], [39.5, 39.5, 40.5, 40.5]],
                     [[40.5, 40.5, 41.5, 41.5], [40.5, 40.5, 41.5, 41.5], [40.5, 40.5, 41.5, 41.5]],
                     [[41.5, 41.5, 42.5, 42.5], [41.5, 41.5, 42.5, 42.5], [41.5, 41.5, 42.5, 42.5]],
                     [[42.5, 42.5, 43.5, 43.5], [42.5, 42.5, 43.5, 43.5], [42.5, 42.5, 43.5, 43.5]]]
        y_bounds = Variable(value=y_corners, name='y_corners', dimensions=['y', 'x', 'cbnds'])
        self.assertEqual(y_bounds.ndim, 3)
        y = Variable(value=y_value, bounds=y_bounds, name='y', dimensions=['y', 'x'])
        suby = y[1:3, 1]
        self.assertEqual(suby.bounds.shape, (2, 1, 4))
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            suby.write(ds)

    def test_system_add_offset_and_scale_factor(self):
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('four', 4)
            var = ds.createVariable('var', int, dimensions=['four'])
            var[:] = [1, 2, 3, 4]
            var.add_offset = 100.
            var.scale_factor = 0.5
        rd = RequestDataset(uri=path)
        sv = SourcedVariable(name='var', request_dataset=rd)
        self.assertEqual(sv.dtype, np.float)
        self.assertTrue(np.all(sv.get_value() == [100.5, 101., 101.5, 102]))
        self.assertNotIn('add_offset', sv.attrs)
        self.assertNotIn('scale_factor', sv.attrs)

    @attr('data', 'cfunits')
    def test_system_conform_units_to(self):
        rd = self.get_request_dataset(conform_units_to='celsius')
        sv = SourcedVariable('tas', request_dataset=rd)[5:9, 5, 9]
        self.assertIsNone(sv._value)
        self.assertEqual(sv.units, 'celsius')
        self.assertLess(sv.get_value().sum(), 200)
        self.assertEqual(sv.units, 'celsius')

    def test_system_get_between_memory(self):
        # Test memory usage is lower when subsetting from file.
        def _get_kb_(dtype, elements):
            nbytes = np.array([1], dtype=dtype).nbytes
            return float((elements * nbytes) / 1024.0)

        var = Variable(name='levels', value=range(100000), dimensions='level')
        original_kb = _get_kb_(var.dtype, var.get_value().size)
        path = self.get_temporary_file_path('foo.nc')
        var.parent.write(path)
        rd = RequestDataset(path)
        field = rd.create_raw_field()
        sub = field['levels'].get_between(5, 100)
        actual_kb = _get_kb_(sub.dtype, sub.get_value().size)
        self.assertLess(actual_kb, original_kb - 100)

    def test_system_using_source_name_from_netcdf(self):
        path = self.get_temporary_file_path('foo.nc')
        with nc_scope(path, mode='w') as ds:
            var = ds.createVariable('name_in_file', 'i1')
            var.defining_attr = 'representative'
        rd = RequestDataset(path)
        sv = SourcedVariable(name='name_in_file', request_dataset=rd)
        sv.set_name('not_in_file')
        self.assertEqual(sv.name, 'not_in_file')
        sv.load()

    def test_fill_value(self):

        var = Variable(name='fill', value=[1, 5, 6], mask=[False, True, False], fill_value=100, dimensions='one')
        path = self.get_temporary_file_path('foo.nc')
        var.write(path)

        rd = RequestDataset(path)
        invar = SourcedVariable(name='fill', request_dataset=rd, protected=True)
        self.assertEqual(invar.fill_value, 100)

    @attr('data')
    def test_get_scatter_slices(self):
        sv = self.get_sourcedvariable(protected=True)
        actual = sv.get_scatter_slices((1, 2, 2))
        desired = ((slice(0, 3650, None), slice(0, 32, None), slice(0, 64, None)),
                   (slice(0, 3650, None), slice(0, 32, None), slice(64, 128, None)),
                   (slice(0, 3650, None), slice(32, 64, None), slice(0, 64, None)),
                   (slice(0, 3650, None), slice(32, 64, None), slice(64, 128, None)))
        self.assertEqual(actual, desired)

    @attr('data')
    def test_getitem(self):
        sv = self.get_sourcedvariable()
        sub = sv[10:20, 5, 6]
        self.assertEqual(sub.shape, (10, 1, 1))
        self.assertIsNone(sub._value)
        self.assertIsNone(sub.dimensions[0].size)
        self.assertEqual(sub.dimensions[0].size_current, 10)

    @attr('data')
    def test_get_dimensions(self):
        sv = self.get_sourcedvariable()
        self.assertTrue(len(sv.dimensions), 3)

    @attr('data')
    def test_get_iter(self):
        sv = self.get_sourcedvariable()[0:10, 0:2, 2:4]
        sv._request_dataset.uid = 5
        itr = sv.get_iter()
        for row in itr:
            self.assertEqual(row[HeaderName.DATASET_IDENTIFER], sv._request_dataset.uid)

    @attr('data')
    def test_get_value(self):
        sv = self.get_sourcedvariable()
        sub = sv[5:11, 3:6, 5:8]
        self.assertEqual(sub.shape, (6, 3, 3))

        with self.nc_scope(self.get_request_dataset().uri, 'r') as ds:
            var = ds.variables[sv.name]
            actual = var[5:11, 3:6, 5:8]

        self.assertNumpyAll(sub.get_value(), actual)

    @attr('data')
    def test_init_from_source(self):
        sv = self.get_sourcedvariable()
        init_from_source(sv)
        self.assertEqual(sv.dtype, np.float32)
        self.assertEqual(sv.fill_value, np.float32(1e20))
        dims = sv.dimensions
        self.assertIsNone(dims[0].size)
        self.assertEqual(dims[0].size_current, 3650)
        self.assertEqual(['time', 'lat', 'lon'], [d.name for d in dims])
        for d in dims:
            # Source indices are always created on dimensions loaded from file.
            self.assertIsNotNone(d.__src_idx__)
        self.assertEqual(sv.attrs['standard_name'], 'air_temperature')

    @attr('data')
    def test_load(self):
        sv = self.get_sourcedvariable()
        self.assertIsNone(sv._value)
        sv.load(cascade=True)
        self.assertIsNotNone(sv._value)

    def test_set_mask(self):
        # Test mask may be set w/out loading source value.
        path = self.get_temporary_file_path('foo.nc')
        vc = VariableCollection()
        var1 = Variable('data', value=np.arange(7), dimensions='dim_data', parent=vc)
        var2_mask = np.zeros(var1.shape, dtype=bool)
        var2_mask[0] = True
        var2 = Variable('data2', value=var1.get_value() + 10, dimensions='dim_data', parent=vc, mask=var2_mask)
        vc.write(path)

        rd = RequestDataset(path)
        vc_ff = rd.create_raw_field()
        svar = vc_ff[var1.name]
        svar2 = vc_ff[var2.name]
        svar.protected = True
        svar2.protected = True
        new_mask = np.zeros(svar.shape[0], dtype=bool)
        new_mask[3] = True
        new_mask[5] = True
        svar.set_mask(new_mask, cascade=True)

        self.assertNumpyAll(svar.get_mask(eager=False), new_mask)

        svar.protected = False
        svar.load()
        self.assertNumpyAll(svar.get_mask(), new_mask)

        svar2.protected = False
        actual = svar2.get_mask(eager=True)
        desired = np.logical_or(new_mask, var2_mask)
        self.assertNumpyAll(actual, desired)

    @attr('data')
    def test_value(self):
        sv = self.get_sourcedvariable()
        sub = sv[5:11, 3:6, 5:8]
        self.assertTrue(sv.get_value().mean() > 0)
        self.assertEqual(sub.get_value().shape, (6, 3, 3))


class TestVariableCollection(AbstractTestInterface):
    def get_variablecollection(self, **kwargs):
        var1 = self.get_variable()
        var2 = self.get_variable(name='wunderbar')

        var3 = Variable(name='lower', value=[[9, 10, 11], [12, 13, 14]], dtype=np.float32, units='large',
                        dimensions=['y', 'x'])

        var4 = Variable(name='coordinate_system', attrs={'proj4': '+proj=latlon'})
        var5 = Variable(name='how_far', value=[5000., 6000., 7000., 8000.], dimensions=['loner'])

        kwargs['variables'] = [var1, var2, var3, var4, var5]
        kwargs['attrs'] = {'foo': 'bar'}

        vc = VariableCollection(**kwargs)
        return vc

    def get_variable(self, name='foo'):
        dim = Dimension('x', size=3)
        value = [4, 5, 6]
        return Variable(name=name, dimensions=dim, value=value)

    def test_init(self):
        var1 = self.get_variable()

        vc = VariableCollection(variables=var1)
        self.assertEqual(len(vc), 1)

        var2 = self.get_variable()
        with self.assertRaises(VariableInCollectionError):
            VariableCollection(variables=[var1, var2])

        var2 = self.get_variable()
        var2._name = 'wunderbar'
        vc = VariableCollection(variables=[var1, var2])
        self.assertEqual(list(vc.keys()), ['foo', 'wunderbar'])

        self.assertEqual(vc.dimensions, {'x': Dimension(name='x', size=3)})

        vc = self.get_variablecollection()
        self.assertEqual(vc.attrs, {'foo': 'bar'})

        # Test with a unique identifier.
        vc = VariableCollection(uid=45)
        self.assertEqual(vc.uid, 45)

    def test_find_by_attribute(self):
        v1 = Variable(attrs={'a': 5}, name='1')
        v2 = Variable(attrs={'b': 5}, name='2')
        vc = VariableCollection(variables=[v1, v2])

        actual = vc.find_by_attribute('c', value=10)
        self.assertEqual(len(actual), 0)

        actual = vc.find_by_attribute('b', 5)
        self.assertEqual(actual[0].name, v2.name)

        pred = lambda x: x == 5
        actual = vc.find_by_attribute(pred=pred)
        self.assertEqual(get_variable_names(actual), (v1.name, v2.name))

    def test_getitem(self):
        vc = self.get_variablecollection()
        slc = {'y': slice(1, 2)}
        sub = vc[slc]
        self.assertEqual(sub['lower'].shape, (1, 3))
        self.assertNotEqual(sub.shapes, vc.shapes)
        self.assertTrue(np.may_share_memory(vc['lower'].get_value(), sub['lower'].get_value()))

    def test_system_as_spatial_collection(self):
        """Test creating a variable collection that is similar to a spatial collection."""

        uid = Variable(name='the_uid', value=[4], dimensions='geom')
        geom = GeometryVariable(name='the_geom', value=[Point(1, 2)], dimensions='geom')
        container = Field(name=uid.get_value()[0], variables=[uid, geom],
                          dimension_map={'geom': {'variable': 'the_geom'}})
        geom.set_ugid(uid)
        coll = VariableCollection()
        coll.add_child(container)
        data = Variable(name='tas', value=[4, 5, 6], dimensions='time')
        contained = Field(name='tas_data', variables=[data])
        coll.children[container.name].add_child(contained)
        self.assertEqual(data.group, [None, 4, 'tas_data'])

    @attr('data')
    def test_system_cf_netcdf(self):
        rd = self.get_request_dataset()
        vc = VariableCollection.read(rd.uri)
        for v in list(vc.values()):
            self.assertIsNone(v._value)
        slc = {'lat': slice(10, 23), 'time': slice(0, 1), 'lon': slice(5, 10)}
        sub = vc['tas'][slc].parent
        self.assertNotEqual(vc.shapes, sub.shapes)
        for v in list(vc.values()):
            self.assertIsNotNone(v.attrs)
            self.assertIsNone(v._value)
            self.assertIsNotNone(v.get_value())

    def test_system_nested(self):
        # Test with nested collections.
        vc = self.get_variablecollection()
        nvc = self.get_variablecollection(name='nest')
        desired = Variable(name='desired', value=[101, 103], dimensions=['one'])
        nvc.add_variable(desired)
        vc.add_child(nvc)
        path = self.get_temporary_file_path('foo.nc')
        vc.write(path)
        # RequestDataset(path).inspect()
        rvc = VariableCollection.read(path)
        self.assertIn('nest', rvc.children)
        self.assertNumpyAll(rvc.children['nest']['desired'].get_value(), desired.get_value())

    def test_system_as_variable_parent(self):
        # Test slicing variables.
        slc1 = {'x': slice(1, 2), 'y': slice(None)}
        slc2 = [slice(None), slice(1, 2)]
        for slc in [slc1, slc2]:
            vc = self.get_variablecollection()
            sub = vc['lower'][slc]
            self.assertNumpyAll(sub.get_value(), np.array([10, 13], dtype=np.float32).reshape(2, 1))
            sub_vc = sub.parent
            self.assertNumpyAll(sub_vc['foo'].get_value(), np.array([5]))
            self.assertNumpyAll(sub_vc['wunderbar'].get_value(), np.array([5]))
            self.assertEqual(sub_vc['how_far'].shape, (4,))
            self.assertNumpyAll(sub_vc['lower'].get_value(), sub.get_value())
            self.assertIn('coordinate_system', sub_vc)

    def test_system_write_netcdf_and_read_netcdf(self):
        vc = self.get_variablecollection()
        path = self.get_temporary_file_path('foo.nc')
        vc.write(path)
        nvc = VariableCollection.read(path)
        path2 = self.get_temporary_file_path('foo2.nc')
        nvc.write(path2)
        self.assertNcEqual(path, path2)

    @attr('data')
    def test_system_write_netcdf_and_read_netcdf_data(self):
        # Test against a real data file.
        rd = self.get_request_dataset()
        rvc = VariableCollection.read(rd.uri)
        self.assertEqual(rvc.dimensions['time'].size_current, 3650)
        for var in rvc.values():
            self.assertIsNone(var._value)
        path3 = self.get_temporary_file_path('foo3.nc')
        rvc.write(path3, dataset_kwargs={'format': rd.metadata['file_format']})
        self.assertNcEqual(path3, rd.uri)

        # Test creating dimensions when writing to netCDF.
        v = Variable(value=np.arange(2 * 4 * 3).reshape(2, 4, 3), name='hello', dimensions=['two', 'four', 'three'])
        path4 = self.get_temporary_file_path('foo4.nc')
        with self.nc_scope(path4, 'w') as ds:
            v.write(ds)
        dname = 'four'
        with self.nc_scope(path4) as ds:
            self.assertIn(dname, ds.dimensions)
        desired = Dimension(dname, 4)
        self.assertEqual(v.dimensions[1], desired)
        vc = VariableCollection.read(path4)
        actual = vc['hello'].dimensions[1]
        actual = Dimension(actual.name, actual.size)
        self.assertEqual(actual, desired)
        path5 = self.get_temporary_file_path('foo5.nc')
        with self.nc_scope(path5, 'w') as ds:
            vc.write(ds)

    def test_add_variable(self):
        vc = VariableCollection()
        var = Variable(name='bounded', value=[2, 3, 4], dtype=float, dimensions='three')
        self.assertIn(var.name, var.parent)
        var.set_extrapolated_bounds('the_bounds', 'bnds')
        self.assertIn('the_bounds', var.parent)
        vc.add_variable(var)
        self.assertIn('the_bounds', vc)

        var1 = Variable('var1', value=np.random.rand(2, 3), dimensions=['two', 'three'])
        var2 = Variable('var2', value=[3, 4, 5, 6, 7], dimensions=['five'])
        vc = VariableCollection(variables=[var1, var2])
        for v in [var1, var2]:
            self.assertIsNotNone(v.dimensions)
            self.assertEqual(id(v.parent), id(vc))
            self.assertIn(v.name, var1.parent)
            self.assertIn(v.name, var2.parent)

    def test_copy(self):
        # Test children added to copied variable collection are not added to the copy source.
        vc = VariableCollection()
        vc_copy = vc.copy()
        vc_copy.add_child(VariableCollection(name='child'))
        self.assertEqual(len(vc.children), 0)

        # Test manipulations on collection variables are not back-propagated.
        vc = VariableCollection()
        var = Variable(name='a', value=[1, 2, 3], dimensions='dima')
        var2 = Variable(name='b', value=[4, 5, 6, 7], dimensions='dimb')
        vc.add_variable(var)
        vc.add_variable(var2)
        vc_copy = vc.copy()
        vc_copy['a'].set_value([4, 5, 6])
        self.assertEqual(var.get_value().tolist(), [1, 2, 3])
        self.assertNotEqual(id(vc['b']), id(vc_copy['b']))
        self.assertNotEqual(id(vc['b'].get_value()), id(vc_copy['b'].get_value()))

    def test_remove_variable(self):
        v1 = Variable(name='vone', value=[1, 2, 3], dimensions='three')
        v1.set_extrapolated_bounds('vone_bounds', 'bounds')
        v2 = Variable(name='vtwo', value=np.arange(6).reshape(3, 2), dimensions=['three', 'two'])
        v3 = Variable(name='vthree')
        vc = VariableCollection(variables=[v1, v2, v3])

        vc.remove_variable(v2)
        self.assertNotIn(v2.name, vc)
        self.assertNotIn('two', vc.dimensions)

        v1 = vc.remove_variable(v1)
        self.assertNotIn(v1.name, vc)
        self.assertNotIn(v1.bounds.name, vc)
        self.assertNotIn(v1.bounds.dimensions[1].name, vc.dimensions)

        v4 = Variable(name='vfour')
        # Test collections on removed variable are not connected.
        v1.parent.add_variable(v4)
        self.assertNotIn(v4.name, vc)

        self.assertEqual(len(vc.dimensions), 0)

    def test_rename_dimension(self):
        parent = VariableCollection()
        var = Variable(name='foo', value=[0], dimensions='one', parent=parent)
        vc = var.parent
        self.assertIsInstance(vc, VariableCollection)
        self.assertEqual(id(parent), id(vc))
        vc.rename_dimension('one', 'one_renamed')
        self.assertEqual(var.dimensions[0].name, 'one_renamed')
