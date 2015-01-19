from collections import OrderedDict
import os
from copy import deepcopy
import numpy as np

from cfunits.cfunits import Units

from ocgis.constants import OCGIS_BOUNDS
from ocgis.interface.base.variable import AbstractSourcedVariable, AbstractValueVariable
from ocgis import constants
from ocgis.exc import EmptySubsetError, ResolutionError, BoundsAlreadyAvailableError
from ocgis.interface.base.dimension.base import VectorDimension, AbstractUidValueDimension, AbstractValueDimension, \
    AbstractDimension, AbstractUidDimension
from ocgis.test.base import TestBase, nc_scope
from ocgis.util.helpers import get_bounds_from_1d
from ocgis.util.itester import itr_products_keywords


class FakeAbstractDimension(AbstractDimension):
    _ndims = None
    _attrs_slice = None


class TestAbstractDimension(TestBase):
    create_dir = False

    @property
    def example_properties(self):
        properties = np.zeros(3, dtype={'names': ['a', 'b'], 'formats': [int, float]})
        properties[0] = (1, 2.5)
        properties[1] = (2, 3.5)
        properties[2] = (3, 4.5)
        return properties

    def test_init(self):
        ad = FakeAbstractDimension()
        self.assertEqual(ad.name, None)

        ad = FakeAbstractDimension(name='time')
        self.assertEqual(ad.name, 'time')

        self.assertEqual(ad.meta, {})

        FakeAbstractDimension(properties=self.example_properties)


class FakeAbstractUidDimension(AbstractUidDimension):
    _attrs_slice = None
    _ndims = 1


class TestAbstractUidDimension(TestBase):
    def test_init(self):
        au = FakeAbstractUidDimension(uid=[1, 2, 3])
        self.assertEqual(au.name_uid, 'None_uid')
        au = FakeAbstractUidDimension(uid=[1, 2, 3], name='foo')
        self.assertEqual(au.name_uid, 'foo_uid')
        self.assertIsNone(au._name_uid)
        au = FakeAbstractUidDimension(uid=[1, 2, 3], name='foo', name_uid='hello')
        self.assertEqual(au.name_uid, 'hello')


class FakeAbstractUidValueDimension(AbstractUidValueDimension):
    _ndims = 1
    _attrs_slice = None

    def _get_value_(self):
        pass


class TestAbstractUidValueDimension(TestBase):

    def test_init(self):
        c = 'celsius'
        ff = FakeAbstractUidValueDimension(conform_units_to=c)
        self.assertEqual(ff.conform_units_to, Units(c))


class FakeAbstractValueDimension(AbstractValueDimension):
    def _get_value_(self):
        pass


class TestAbstractValueDimension(TestBase):
    create_dir = False

    def test_init(self):
        FakeAbstractValueDimension()
        self.assertEqual(AbstractValueDimension.__bases__, (AbstractValueVariable,))

    def test_name_value(self):
        name_value = 'foo'
        avd = FakeAbstractValueDimension(name_value=name_value)
        self.assertEqual(avd.name_value, name_value)

        name = 'foobar'
        avd = FakeAbstractValueDimension(name=name)
        self.assertEqual(avd.name_value, name)
        self.assertIsNone(avd._name_value)


class TestVectorDimension(TestBase):
    def test_init(self):
        vd = VectorDimension(value=[4, 5])
        self.assertIsInstances(vd, (AbstractSourcedVariable, AbstractUidValueDimension))
        self.assertIsInstance(vd.attrs, OrderedDict)
        self.assertIsNone(vd.name)
        self.assertIsNone(vd.name_value)
        self.assertEqual(vd.name_uid, 'None_uid')
        self.assertEqual(vd.name_bounds, 'None_{0}'.format(OCGIS_BOUNDS))
        self.assertIsNone(vd.axis)
        self.assertEqual(vd.name_bounds_dimension, OCGIS_BOUNDS)

        # test passing attributes to the constructor
        attrs = {'something': 'underground'}
        vd = VectorDimension(value=[4, 5], attrs=attrs, axis='D', name_bounds_dimension='vds')
        self.assertEqual(vd.attrs, attrs)
        self.assertEqual(vd.axis, 'D')
        self.assertEqual(vd.name_bounds_dimension, 'vds')

        # empty dimensions are not allowed
        with self.assertRaises(ValueError):
            VectorDimension()

        # test passing the name bounds
        vd = VectorDimension(value=[5, 6], name_bounds='lat_bnds')
        self.assertEqual(vd.name_bounds, 'lat_bnds')

    def test_init_conform_units_to(self):
        target = np.array([4, 5, 6])
        target_copy = target.copy()
        vd = VectorDimension(value=target, units='celsius', conform_units_to='kelvin')
        self.assertNumpyNotAll(vd.value, target_copy)
        self.assertNumpyAll(vd.value, np.array([277.15, 278.15, 279.15]))
        self.assertEqual(vd.units, 'kelvin')
        self.assertEqual(vd.cfunits, Units('kelvin'))

        target = np.array([4., 5., 6.])
        target_bounds = np.array([[3.5, 4.5], [4.5, 5.5], [5.5, 6.5]])
        vd = VectorDimension(value=target, bounds=target_bounds, units='celsius', conform_units_to='kelvin')
        self.assertNumpyAll(vd.bounds, np.array([[276.65, 277.65], [277.65, 278.65], [278.65, 279.65]]))

    def test_bad_dtypes(self):
        vd = VectorDimension(value=181.5, bounds=[181, 182])
        self.assertEqual(vd.value.dtype, vd.bounds.dtype)

        with self.assertRaises(ValueError):
            VectorDimension(value=181.5, bounds=['a', 'b'])

    def test_bad_keywords(self):
        # there should be keyword checks on the bad keywords names
        with self.assertRaises(ValueError):
            VectorDimension(value=40, bounds=[38, 42], ddtype=float)

    def test_boolean_slice(self):
        """Test slicing with boolean values."""

        vdim = VectorDimension(value=[4, 5, 6], bounds=[[3, 5], [4, 6], [5, 7]])
        vdim_slc = vdim[np.array([True, False, True])]
        self.assertFalse(len(vdim_slc) > 2)
        self.assertNumpyAll(vdim_slc.value, np.array([4, 6]))
        self.assertNumpyAll(vdim_slc.bounds, np.array([[3, 5], [5, 7]]))

    def test_bounds_only_two_dimensional(self):
        value = [10, 20, 30, 40, 50]
        bounds = [
            [[b - 5, b + 5, b + 10] for b in value],
            value,
            5
        ]
        for b in bounds:
            with self.assertRaises(ValueError):
                VectorDimension(value=value, bounds=b)

    def test_dtype(self):
        value = [10, 20, 30, 40, 50]
        vdim = VectorDimension(value=value)
        self.assertEqual(vdim.dtype, np.array(value).dtype)

    def test_get_iter(self):
        vdim = VectorDimension(value=[10, 20, 30, 40, 50])
        with self.assertRaises(ValueError):
            list(vdim.get_iter())

        vdim = VectorDimension(value=[10, 20, 30, 40, 50], name='foo')
        tt = list(vdim.get_iter())
        self.assertEqual(tt[3], (3, {'foo_uid': 4, 'foo': 40, 'lb_foo': None, 'ub_foo': None}))
        self.assertIsInstance(tt[0][1], OrderedDict)

        vdim = VectorDimension(value=[10, 20, 30, 40, 50], bounds=[(ii - 5, ii + 5) for ii in [10, 20, 30, 40, 50]],
                               name='foo', name_uid='hi')
        tt = list(vdim.get_iter())
        self.assertEqual(tt[3], (3, {'hi': 4, 'foo': 40, 'lb_foo': 35, 'ub_foo': 45}))

        vdim = VectorDimension(value=[4, 5, 6, 7, 8, 9, 10], name='new')
        for slc, row in vdim.get_iter(with_bounds=False):
            for k in row.iterkeys():
                self.assertFalse(OCGIS_BOUNDS in k)

    def test_interpolate_bounds(self):
        value = [10, 20, 30, 40, 50]

        vdim = VectorDimension(value=value)
        self.assertEqual(vdim.bounds, None)

        vdim = VectorDimension(value=value)
        vdim.set_extrapolated_bounds()
        self.assertEqual(vdim.bounds.tostring(),
                         '\x05\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00#\x00\x00\x00\x00\x00\x00\x00#\x00\x00\x00\x00\x00\x00\x00-\x00\x00\x00\x00\x00\x00\x00-\x00\x00\x00\x00\x00\x00\x007\x00\x00\x00\x00\x00\x00\x00')

    def test_load_from_source(self):
        """Test loading from a fake data source."""

        vdim = VectorDimension(src_idx=[0, 1, 2, 3], data='foo')
        self.assertNumpyAll(vdim.uid, np.array([1, 2, 3, 4], dtype=constants.NP_INT))
        with self.assertRaises(NotImplementedError):
            vdim.value
        with self.assertRaises(NotImplementedError):
            vdim.resolution

    def test_name_bounds(self):
        vd = VectorDimension(value=[5, 6], name='hello')
        self.assertEqual(vd.name_bounds, 'hello_bounds')
        self.assertIsNone(vd._name_bounds)

        vd = VectorDimension(value=[5, 6], name_bounds='hello')
        self.assertEqual(vd.name_bounds, 'hello')
        self.assertEqual(vd._name_bounds, 'hello')
        self.assertIsNone(vd.name)

        vd = VectorDimension(value=[5, 6], name_bounds='hello', name='hi')
        self.assertEqual(vd.name_bounds, 'hello')
        self.assertEqual(vd._name_bounds, 'hello')
        self.assertEqual(vd.name, 'hi')

        vd = VectorDimension(value=[5, 6], name='hello', name_bounds_dimension='whatever')
        self.assertEqual(vd.name_bounds, 'hello_whatever')

    def test_name_bounds_tuple(self):
        vd = VectorDimension(value=[4, 5])
        self.assertEqual(vd.name_bounds_tuple, ('lb_None', 'ub_None'))
        self.assertIsNone(vd._name_bounds_tuple)

        vd = VectorDimension(value=[4, 5], name='never')
        self.assertEqual(vd.name_bounds_tuple, ('lb_never', 'ub_never'))

        vd = VectorDimension(value=[4, 5], name_bounds_tuple=('a', 'b'))
        self.assertEqual(vd.name_bounds_tuple, ('a', 'b'))

    def test_one_value(self):
        """Test passing a single value."""

        values = [5, np.array([5])]
        for value in values:
            vdim = VectorDimension(value=value, src_idx=10)
            self.assertEqual(vdim.value[0], 5)
            self.assertEqual(vdim.uid[0], 1)
            self.assertEqual(len(vdim.uid), 1)
            self.assertEqual(vdim.shape, (1,))
            self.assertIsNone(vdim.bounds)
            self.assertEqual(vdim[0].value[0], 5)
            self.assertEqual(vdim[0].uid[0], 1)
            self.assertEqual(vdim[0]._src_idx[0], 10)
            self.assertIsNone(vdim[0].bounds)
            with self.assertRaises(ResolutionError):
                vdim.resolution

    def test_resolution_with_units(self):
        vdim = VectorDimension(value=[5, 10, 15], units='large')
        self.assertEqual(vdim.resolution, 5.0)

    def test_set_extrapolated_bounds(self):
        value = np.array([1, 2, 3, 4], dtype=float)
        vd = VectorDimension(value=value)
        self.assertIsNone(vd.bounds)
        self.assertFalse(vd._has_interpolated_bounds)
        vd.set_extrapolated_bounds()
        self.assertTrue(vd._has_interpolated_bounds)
        actual = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5]], dtype=float)
        self.assertNumpyAll(vd.bounds, actual)

        # attempt to extrapolate when the bound are already present
        value = np.array([1.5])
        bounds = np.array([[1.0, 2.0]])
        vd = VectorDimension(value=value, bounds=bounds)
        self.assertFalse(vd._has_interpolated_bounds)
        with self.assertRaises(BoundsAlreadyAvailableError):
            vd.set_extrapolated_bounds()

    def test_set_reference(self):
        """Test setting values on the internal value array using indexing."""

        vdim = VectorDimension(value=[4, 5, 6])
        vdim_slc = vdim[1]
        self.assertEqual(vdim_slc.uid[0], 2)
        vdim_slc2 = vdim[:]
        self.assertNumpyAll(vdim_slc2.value, vdim.value)
        vdim._value[1] = 500
        self.assertNumpyAll(vdim.value, np.array([4, 500, 6]))
        with self.assertRaises(TypeError):
            vdim.bounds[1, :]
        self.assertNumpyAll(vdim.value, vdim_slc2.value)
        vdim_slc2._value[2] = 1000
        self.assertNumpyAll(vdim.value, vdim_slc2.value)

    def test_slice_source_idx_only(self):
        vdim = VectorDimension(src_idx=[4, 5, 6], data='foo')
        vdim_slice = vdim[0]
        self.assertEqual(vdim_slice._src_idx[0], 4)

    def test_units_with_bounds(self):
        value = [5., 10., 15.]
        vdim = VectorDimension(value=value, units='celsius',
                               bounds=get_bounds_from_1d(np.array(value)))
        vdim.cfunits_conform(Units('kelvin'))
        self.assertNumpyAll(vdim.bounds, np.array([[275.65, 280.65], [280.65, 285.65], [285.65, 290.65]]))

    def test_with_bounds(self):
        """Test passing bounds to the constructor."""

        vdim = VectorDimension(value=[4, 5, 6], bounds=[[3, 5], [4, 6], [5, 7]])
        self.assertNumpyAll(vdim.bounds, np.array([[3, 5], [4, 6], [5, 7]]))
        self.assertNumpyAll(vdim.uid, np.array([1, 2, 3], dtype=constants.NP_INT))
        self.assertEqual(vdim.resolution, 2.0)

    def test_with_units(self):
        vdim = VectorDimension(value=[5, 10, 15], units='celsius')
        self.assertEqual(vdim.cfunits, Units('celsius'))
        vdim.cfunits_conform(Units('kelvin'))
        self.assertNumpyAll(vdim.value, np.array([278.15, 283.15, 288.15]))

    def test_with_units_and_bounds_convert_after_load(self):
        vdim = VectorDimension(value=[5., 10., 15.], units='celsius')
        vdim.set_extrapolated_bounds()
        vdim.cfunits_conform(Units('kelvin'))
        self.assertNumpyAll(vdim.bounds, np.array([[275.65, 280.65], [280.65, 285.65], [285.65, 290.65]]))

    def test_with_units_and_bounds_interpolation(self):
        vdim = VectorDimension(value=[5., 10., 15.], units='celsius')
        vdim.set_extrapolated_bounds()
        vdim.cfunits_conform(Units('kelvin'))
        self.assertNumpyAll(vdim.bounds, np.array([[275.65, 280.65], [280.65, 285.65], [285.65, 290.65]]))

    def test_write_to_netcdf_dataset(self):
        path = os.path.join(self.current_dir_output, 'foo.nc')

        other_bounds_name = 'bnds'
        keywords = dict(with_bounds=[True, False],
                        with_attrs=[True, False],
                        unlimited=[False, True],
                        kwargs=[{}, {'zlib': True}],
                        bounds_dimension_name=[None, other_bounds_name],
                        axis=[None, 'GG'],
                        name=[None, 'temporal'],
                        name_bounds=[None, 'time_bounds'],
                        name_value=[None, 'time'],
                        format=[None, 'NETCDF4_CLASSIC'])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            if k.with_attrs:
                attrs = {'a': 5, 'b': np.array([5, 6])}
            else:
                attrs = None
            vd = VectorDimension(value=[2., 4.], attrs=attrs, name=k.name, name_bounds=k.name_bounds,
                                 name_value=k.name_value, axis=k.axis)
            if k.with_bounds:
                vd.set_extrapolated_bounds()
            with nc_scope(path, 'w') as ds:
                try:
                    vd.write_to_netcdf_dataset(ds, unlimited=k.unlimited, bounds_dimension_name=k.bounds_dimension_name,
                                               **k.kwargs)
                except ValueError:
                    self.assertIsNone(vd.name)
                    continue

            with nc_scope(path, 'r') as ds:
                var = ds.variables[vd.name_value]

                if k.axis is None:
                    axis_actual = ''
                else:
                    axis_actual = vd.axis
                self.assertEqual(var.axis, axis_actual)

                try:
                    self.assertIn(constants.OCGIS_BOUNDS, ds.dimensions)
                except AssertionError:
                    try:
                        self.assertFalse(k.with_bounds)
                    except AssertionError:
                        try:
                            self.assertEqual(k.bounds_dimension_name, other_bounds_name)
                        except AssertionError:
                            self.assertIsNotNone(k.name_bounds_suffix)
                            self.assertIsNone(k.bounds_dimension_name)
                            self.assertIn(k.name_bounds_suffix, ds.variables[vd.name_bounds].dimensions)
                try:
                    self.assertFalse(ds.dimensions[vd.name].isunlimited())
                except AssertionError:
                    self.assertTrue(k.unlimited)

                try:
                    self.assertEqual(var.a, attrs['a'])
                    self.assertNumpyAll(var.b, attrs['b'])
                except AttributeError:
                    self.assertFalse(k.with_attrs)
                try:
                    self.assertEqual(var.bounds, vd.name_bounds)
                    self.assertNumpyAll(vd.bounds, ds.variables[vd.name_bounds][:])
                except (AttributeError, KeyError):
                    self.assertFalse(k.with_bounds)
                self.assertEqual(var._name, vd.name_value)
                self.assertEqual(var.dimensions, (vd.name,))
                self.assertNumpyAll(vd.value, var[:])

    def test_write_to_netcdf_dataset_bounds_dimension_exists(self):
        """Test writing with bounds when the bounds dimension has already been created."""

        vd = VectorDimension(value=[3., 7.], name='one')
        vd.set_extrapolated_bounds()
        vd2 = VectorDimension(value=[5., 6.], name='two')
        vd2.set_extrapolated_bounds()
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            vd.write_to_netcdf_dataset(ds)
            vd2.write_to_netcdf_dataset(ds)
            self.assertEqual(ds.variables.keys(), ['one', 'one_bounds', 'two', 'two_bounds'])

    def test_get_between(self):
        vdim = VectorDimension(value=[0])
        with self.assertRaises(EmptySubsetError):
            vdim.get_between(100, 200)

        vdim = VectorDimension(value=[100, 200, 300, 400])
        vdim_between = vdim.get_between(100, 200)
        self.assertEqual(len(vdim_between), 2)

    def test_get_between_bounds(self):
        value = [0., 5., 10.]
        bounds = [[-2.5, 2.5], [2.5, 7.5], [7.5, 12.5]]

        # # a reversed copy of these bounds are created here
        value_reverse = deepcopy(value)
        value_reverse.reverse()
        bounds_reverse = deepcopy(bounds)
        bounds_reverse.reverse()
        for ii in range(len(bounds)):
            bounds_reverse[ii].reverse()

        data = {'original': {'value': value, 'bounds': bounds},
                'reversed': {'value': value_reverse, 'bounds': bounds_reverse}}
        for key in ['original', 'reversed']:
            vdim = VectorDimension(value=data[key]['value'],
                                   bounds=data[key]['bounds'])

            vdim_between = vdim.get_between(1, 3)
            self.assertEqual(len(vdim_between), 2)
            if key == 'original':
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x04\xc0\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x1e@')
            else:
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04\xc0')
            self.assertEqual(vdim.resolution, 5.0)

            ## preference is given to the lower bound in the case of "ties" where
            ## the value could be assumed part of the lower or upper cell
            vdim_between = vdim.get_between(2.5, 2.5)
            self.assertEqual(len(vdim_between), 1)
            if key == 'original':
                self.assertNumpyAll(vdim_between.bounds, np.array([[2.5, 7.5]]))
            else:
                self.assertNumpyAll(vdim_between.bounds, np.array([[7.5, 2.5]]))

            ## if the interval is closed and the subset range falls only on bounds
            ## value then the subset will be empty
            with self.assertRaises(EmptySubsetError):
                vdim.get_between(2.5, 2.5, closed=True)

            vdim_between = vdim.get_between(2.5, 7.5)
            if key == 'original':
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00)@')
            else:
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00)@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x04@')

    def test_get_between_use_bounds(self):
        value = [3., 5.]
        bounds = [[2., 4.], [4., 6.]]
        vdim = VectorDimension(value=value, bounds=bounds)
        ret = vdim.get_between(3, 4.5, use_bounds=False)
        self.assertNumpyAll(ret.value, np.array([3.]))
        self.assertNumpyAll(ret.bounds, np.array([[2., 4.]]))
