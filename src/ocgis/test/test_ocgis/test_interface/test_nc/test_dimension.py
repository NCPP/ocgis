from ocgis import VectorDimension
from ocgis.interface.nc.dimension import NcVectorDimension
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class TestNcVectorDimension(TestBase):
    def get(self, **kwargs):
        rd = self.get_request_dataset()
        src_idx = self.get_src_idx()
        kwargs['request_dataset'] = kwargs.pop('request_dataset', rd)
        kwargs['src_idx'] = src_idx
        kwargs['name'] = 'time'

        ret = NcVectorDimension(**kwargs)
        return ret

    def get_from_source(self, name):
        with self.nc_scope(self.get_request_dataset().uri) as ds:
            var = ds.variables[name]
            return var[self.get_src_idx(), ...]

    def get_request_dataset(self):
        return self.test_data.get_rd('cancm4_tas')

    def get_src_idx(self):
        src_idx = [1, 2, 3, 4, 5]
        return src_idx

    @attr('data')
    def test_init(self):
        self.assertEqual(NcVectorDimension.__bases__, (VectorDimension,))
        self.assertIsInstance(self.get(), NcVectorDimension)

    @attr('data')
    def test_bounds(self):
        vdim = self.get(axis='T')
        self.assertEqual(vdim.bounds.shape[0], vdim._src_idx.shape[0])
        self.assertNumpyAll(vdim.bounds, self.get_from_source('time_bnds'))

        # Test bounds may be none.
        vdim = self.get(request_dataset=None, value=5)
        self.assertIsNone(vdim._request_dataset)
        self.assertIsNone(vdim.bounds)

    @attr('data')
    def test_value(self):
        vdim = self.get(axis='T')
        self.assertEqual(vdim.value.shape, vdim._src_idx.shape)
        self.assertNumpyAll(vdim.value, self.get_from_source('time'))
