import os
from copy import deepcopy
from itertools import izip

import fiona
import numpy as np

import ocgis
from ocgis import constants
from ocgis.api.operations import OcgOperations
from ocgis.exc import DefinitionValidationError
from ocgis.interface.base.crs import CFWGS84, CoordinateReferenceSystem
from ocgis.test.base import TestBase, attr
from ocgis.util.geom_cabinet import GeomCabinetIterator


class Test(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.maurer = self.test_data.get_rd('maurer_bccr_1950')
        self.cancm4 = self.test_data.get_rd('cancm4_tasmax_2001')
        self.tasmin = self.test_data.get_rd('cancm4_tasmin_2001')

    @property
    def california(self):
        ret = list(GeomCabinetIterator('state_boundaries', select_uid=[25]))
        return ret

    @property
    def dataset(self):
        dataset = [deepcopy(self.maurer), deepcopy(self.cancm4)]
        return dataset

    def get_ops(self, kwds={}):
        geom = self.california
        ops = OcgOperations(dataset=self.dataset, snippet=True, geom=geom, output_format='numpy')
        for k, v in kwds.iteritems():
            setattr(ops, k, v)
        return ops

    def get_ref(self, kwds={}):
        ops = self.get_ops(kwds=kwds)
        ret = ops.execute()
        return ret[25]

    @attr('data')
    def test_default(self):
        ops = self.get_ops()
        ret = ops.execute()

        self.assertEqual(set(['Prcp', 'tasmax']), set(ret[25].keys()))

        shapes = {'Prcp': (1, 1, 1, 77, 83), 'tasmax': (1, 1, 1, 5, 4)}
        for (ugid, field_alias, var_alias, variable), shape in izip(ret.get_iter_elements(), shapes):
            self.assertEqual(variable.value.shape, shapes[var_alias])

    @attr('data')
    def test_vector_wrap(self):
        geom = self.california
        keys = [['maurer_bccr_1950', (1, 12, 1, 77, 83)], ['cancm4_tasmax_2011', (1, 3650, 1, 5, 4)]]
        for key in keys:
            prev_value = None
            for vector_wrap in [True, False]:
                rd = self.test_data.get_rd(key[0])
                prefix = 'vw_{0}_{1}'.format(vector_wrap, rd.variable)
                ops = ocgis.OcgOperations(dataset=rd, geom=geom, snippet=False, vector_wrap=vector_wrap, prefix=prefix)
                ret = ops.execute()
                ref = ret.gvu(25, rd.variable)
                self.assertEqual(ref.shape, key[1])
                if prev_value is None:
                    prev_value = ref
                else:
                    self.assertTrue(np.all(ref == prev_value))

    @attr('data')
    def test_aggregate_clip(self):
        kwds = {'aggregate': True, 'spatial_operation': 'clip'}

        ref = self.get_ref(kwds)
        for field in ref.values():
            for variable in field.variables.values():
                self.assertEqual(field.spatial.geom.shape, (1, 1))
                self.assertEqual(variable.value.shape, (1, 1, 1, 1, 1))

    @attr('data')
    def test_calculation(self):
        calc = [{'func': 'mean', 'name': 'mean'}, {'func': 'std', 'name': 'std'}]
        calc_grouping = ['year']
        kwds = {'aggregate': True,
                'spatial_operation': 'clip',
                'calc': calc,
                'calc_grouping': calc_grouping,
                'output_format': constants.OUTPUT_FORMAT_NUMPY,
                'geom': self.california,
                'dataset': self.dataset,
                'snippet': False}
        ops = OcgOperations(**kwds)
        ret = ops.execute()

        ref = ret[25]['Prcp']
        self.assertEquals(set(ref.variables.keys()), set(['mean', 'std']))
        for value in ref.variables.itervalues():
            self.assertEqual(value.value.shape, (1, 1, 1, 1, 1))
        ref = ret[25]['tasmax']
        self.assertEquals(set(ref.variables.keys()), set(['mean', 'std']))
        for value in ref.variables.itervalues():
            self.assertEqual(value.value.shape, (1, 10, 1, 1, 1))

    @attr('data')
    def test_same_variable_name(self):
        ds = deepcopy([self.cancm4, self.cancm4])

        with self.assertRaises(KeyError):
            OcgOperations(dataset=ds)
        ds[0].alias = 'foo'
        ds[1].alias = 'foo'
        with self.assertRaises(KeyError):
            OcgOperations(dataset=ds)

        ds = [deepcopy(self.cancm4), deepcopy(self.cancm4)]
        ds[0].alias = 'foo_var'
        ops = OcgOperations(dataset=ds, snippet=True)
        ret = ops.execute()
        self.assertEqual(set(ret[1].keys()), set(['foo_var', 'tasmax']))
        values = [v.variables[k] for k, v in ret[1].iteritems()]
        self.assertTrue(np.all(values[0].value == values[1].value))

    @attr('slow')
    def test_consolidating_projections(self):

        def assert_projection(path, check_ugid=True):
            try:
                source = [fiona.open(path, 'r')]
            except fiona.errors.DriverError:
                shp_path = os.path.split(path)[0]
                prefix = os.path.split(shp_path)[1]
                shp_path = os.path.join(shp_path, 'shp')
                if check_ugid:
                    ids = ['gid', 'ugid']
                else:
                    ids = ['gid']
                source = [fiona.open(os.path.join(shp_path, prefix + '_' + suffix + '.shp')) for suffix in ids]

            try:
                for src in source:
                    self.assertEqual(CoordinateReferenceSystem(value=src.meta['crs']), CFWGS84())
            finally:
                for src in source:
                    src.close()

        rd1 = self.test_data.get_rd('narccap_rcm3')
        rd1.alias = 'rcm3'
        rd2 = self.test_data.get_rd('narccap_crcm')
        rd2.alias = 'crcm'
        rd = [rd1, rd2]

        for output_format in [constants.OUTPUT_FORMAT_CSV_SHAPEFILE, constants.OUTPUT_FORMAT_SHAPEFILE,
                              constants.OUTPUT_FORMAT_NETCDF]:

            try:
                ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format=output_format,
                                          geom='state_boundaries', agg_selection=False, select_ugid=[25],
                                          prefix='ca' + output_format, output_crs=CFWGS84())
                ret = ops.execute()
            # writing to a reference projection is currently not supported for netCDF data.
            except DefinitionValidationError:
                if output_format == constants.OUTPUT_FORMAT_NETCDF:
                    continue
                else:
                    raise
            assert_projection(ret)

            ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format=output_format, geom='state_boundaries',
                                      agg_selection=True, prefix='states' + output_format, output_crs=CFWGS84())
            ret = ops.execute()
            assert_projection(ret)

            ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format=output_format,
                                      prefix='rcm3_crcm_domain' + output_format, output_crs=CFWGS84())
            ret = ops.execute()
            assert_projection(ret, check_ugid=False)
