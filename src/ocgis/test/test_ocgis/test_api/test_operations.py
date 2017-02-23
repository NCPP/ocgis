import csv
import datetime
import itertools
import os
from datetime import datetime as dt
from types import FunctionType

import numpy as np
from numpy import dtype

import ocgis
from ocgis import Field
from ocgis import constants
from ocgis import env
from ocgis.api.operations import OcgOperations
from ocgis.api.parms import definition
from ocgis.api.parms.definition import RegridOptions, OutputFormat, SpatialWrapping
from ocgis.api.request.base import RequestDataset
from ocgis.constants import WrappedState
from ocgis.exc import DefinitionValidationError, DimensionNotFound, RequestValidationError
from ocgis.interface.base.crs import CFWGS84, Spherical, CoordinateReferenceSystem
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, SpatialDimension
from ocgis.interface.base.variable import Variable
from ocgis.test.base import TestBase, attr
from ocgis.test.test_simple.test_dependencies import create_mftime_nc_files
from ocgis.util.geom_cabinet import GeomCabinetIterator, GeomCabinet
from ocgis.util.helpers import make_poly


class TestOcgOperations(TestBase):
    def setUp(self):
        TestBase.setUp(self)

        rds = [self.test_data.get_rd('cancm4_tasmin_2001'), self.test_data.get_rd('cancm4_tasmax_2011'),
               self.test_data.get_rd('cancm4_tas')]

        time_range = [dt(2000, 1, 1), dt(2000, 12, 31)]
        level_range = [2, 2]

        self.datasets = [{'uri': rd.uri, 'variable': rd.variable, 'time_range': time_range, 'level_range': level_range}
                         for rd in rds]
        self.datasets_no_range = [{'uri': rd.uri, 'variable': rd.variable} for rd in rds]

    @attr('data')
    def test_init(self):
        with self.assertRaises(DefinitionValidationError):
            OcgOperations()
        ops = OcgOperations(dataset=self.datasets)
        self.assertEqual(ops.regrid_destination, None)
        self.assertDictEqual(ops.regrid_options, RegridOptions.default)
        self.assertIsNone(ops.geom_select_uid)
        self.assertIsNone(ops.geom_uid)

        self.assertFalse(ops.melted)
        env.MELTED = True
        ops = OcgOperations(dataset=self.datasets)
        self.assertEqual(ops.melted, env.MELTED)
        self.assertTrue(ops.melted)

        ops = OcgOperations(dataset=self.datasets, geom_select_uid=[4, 5], select_ugid=[5, 6, 7])
        self.assertEqual(ops.geom_select_uid, (4, 5))

        ops = OcgOperations(dataset=self.datasets, geom_uid='ID')
        self.assertEqual(ops.geom_uid, 'ID')
        geom = ops._get_object_('geom')
        self.assertEqual(geom.geom_uid, 'ID')

        s = 'STATE_NAME in ("Wisconsin", "Vermont")'
        ops = OcgOperations(dataset=self.datasets, geom_select_sql_where=s, geom='state_boundaries')
        self.assertEqual(ops.geom_select_sql_where, s)
        self.assertEqual(len(ops.geom), 2)

    @attr('data')
    def test_str(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        ret = str(ops)
        self.assertTrue(str(ret).startswith('OcgOperations'))
        self.assertGreater(len(ret), 1000)

    @attr('data')
    def test_system_scalar_level_dimension(self):
        """Test scalar level dimensions are not dropped in netCDF output."""

        rd = self.test_data.get_rd('cancm4_tas')
        desired_height_metadata = rd.source_metadata['variables']['height']
        ops = OcgOperations(dataset=rd, output_format='nc', snippet=True)
        ret = ops.execute()

        rd_out = RequestDataset(uri=ret)
        actual = rd_out.source_metadata['variables']['height']

        # Not worried about order of attributes.
        desired_height_metadata['attrs'] = dict(desired_height_metadata['attrs'])
        actual['attrs'] = dict(actual['attrs'])

        self.assertDictEqual(actual, desired_height_metadata)

    @attr('data')
    def test_get_base_request_size(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        size = ops.get_base_request_size()
        self.assertEqual(size, {'variables': {'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                      'temporal': {'kb': 28.515625, 'shape': (3650,),
                                                                   'dtype': dtype('float64')},
                                                      'value': {'kb': 116800.0, 'shape': (1, 3650, 1, 64, 128),
                                                                'dtype': dtype('float32')},
                                                      'realization': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                      'col': {'kb': 1.0, 'shape': (128,), 'dtype': dtype('float64')},
                                                      'row': {'kb': 0.5, 'shape': (64,), 'dtype': dtype('float64')}}},
                                'total': 116830.015625})

        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, regrid_destination=rd).get_base_request_size()

    @attr('data')
    def test_get_base_request_size_multifile(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        rds = [rd1, rd2]
        ops = OcgOperations(dataset=rds)
        size = ops.get_base_request_size()
        self.assertEqual({'variables': {'pr': {'level': {'kb': 0.0, 'shape': None, 'dtype': None},
                                               'temporal': {'kb': 228.25, 'shape': (29216,), 'dtype': dtype('float64')},
                                               'value': {'kb': 1666909.75, 'shape': (1, 29216, 1, 109, 134),
                                                         'dtype': dtype('float32')},
                                               'realization': {'kb': 0.0, 'shape': None, 'dtype': None},
                                               'col': {'kb': 1.046875, 'shape': (134,), 'dtype': dtype('float64')},
                                               'row': {'kb': 0.8515625, 'shape': (109,), 'dtype': dtype('float64')}},
                                        'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                'temporal': {'kb': 28.515625, 'shape': (3650,),
                                                             'dtype': dtype('float64')},
                                                'value': {'kb': 116800.0, 'shape': (1, 3650, 1, 64, 128),
                                                          'dtype': dtype('float32')},
                                                'realization': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                'col': {'kb': 1.0, 'shape': (128,), 'dtype': dtype('float64')},
                                                'row': {'kb': 0.5, 'shape': (64,), 'dtype': dtype('float64')}}},
                          'total': 1783969.9140625}, size)

    @attr('data')
    def test_get_base_request_size_multifile_with_geom(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        rds = [rd1, rd2]
        ops = OcgOperations(dataset=rds, geom='state_boundaries', select_ugid=[23])
        size = ops.get_base_request_size()
        self.assertEqual(size, {'variables': {'pr': {'level': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                     'temporal': {'kb': 228.25, 'shape': (29216,),
                                                                  'dtype': dtype('float64')},
                                                     'value': {'kb': 21341.375, 'shape': (1, 29216, 1, 17, 11),
                                                               'dtype': dtype('float32')},
                                                     'realization': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                     'col': {'kb': 0.0859375, 'shape': (11,),
                                                             'dtype': dtype('float64')},
                                                     'row': {'kb': 0.1328125, 'shape': (17,),
                                                             'dtype': dtype('float64')}},
                                              'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                      'temporal': {'kb': 28.515625, 'shape': (3650,),
                                                                   'dtype': dtype('float64')},
                                                      'value': {'kb': 171.09375, 'shape': (1, 3650, 1, 4, 3),
                                                                'dtype': dtype('float32')},
                                                      'realization': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                      'col': {'kb': 0.0234375, 'shape': (3,),
                                                              'dtype': dtype('float64')},
                                                      'row': {'kb': 0.03125, 'shape': (4,),
                                                              'dtype': dtype('float64')}}}, 'total': 21769.5078125})

    @attr('data')
    def test_get_base_request_size_test_data(self):
        for key in self.test_data.keys():
            rd = self.test_data.get_rd(key)
            try:
                ops = OcgOperations(dataset=rd)
            # the project cmip data may raise an exception since projection is not associated with a variable
            except DimensionNotFound:
                rd = self.test_data.get_rd(key, kwds=dict(
                    dimension_map={'R': 'projection', 'T': 'time', 'X': 'longitude', 'Y': 'latitude'}))
                ops = OcgOperations(dataset=rd)
            ret = ops.get_base_request_size()
            self.assertTrue(ret['total'] > 1)

    @attr('data')
    def test_get_base_request_size_with_calculation(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}],
                            calc_grouping=['month'])
        size = ops.get_base_request_size()
        self.assertEqual(size['variables']['tas']['temporal']['shape'][0], 3650)

    @attr('data')
    def test_get_base_request_size_with_geom(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23])
        size = ops.get_base_request_size()
        self.assertEqual(size, {'variables': {'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                      'temporal': {'kb': 28.515625, 'shape': (3650,),
                                                                   'dtype': dtype('float64')},
                                                      'value': {'kb': 171.09375, 'shape': (1, 3650, 1, 4, 3),
                                                                'dtype': dtype('float32')},
                                                      'realization': {'kb': 0.0, 'shape': None, 'dtype': None},
                                                      'col': {'kb': 0.0234375, 'shape': (3,),
                                                              'dtype': dtype('float64')},
                                                      'row': {'kb': 0.03125, 'shape': (4,),
                                                              'dtype': dtype('float64')}}}, 'total': 199.6640625})

    @attr('data')
    def test_get_meta(self):
        ops = OcgOperations(dataset=self.datasets)
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)

        ops = OcgOperations(dataset=self.datasets, calc=[{'func': 'mean', 'name': 'my_mean'}],
                            calc_grouping=['month'])
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)

    @attr('data')
    def test_keyword_abstraction(self):
        kk = definition.Abstraction

        k = kk()
        self.assertEqual(k.value, None)
        self.assertEqual(str(k), 'abstraction="None"')

        k = kk('point')
        self.assertEqual(k.value, 'point')

        with self.assertRaises(DefinitionValidationError):
            kk('pt')

    @attr('data')
    def test_keyword_aggregate(self):
        rd = self.test_data.get_rd('rotated_pole_cnrm_cerfacs')

        ofield = rd.get()[:, 0:10, :, 0:10, 0:10]
        ovalue = ofield.variables['pr'].value
        manual_mean = ovalue[0, 4, 0, :, :].mean()

        slc = [None, [0, 10], None, [0, 10], [0, 10]]
        for output_format in ['numpy', 'csv']:
            ops = OcgOperations(dataset=rd, output_format=output_format, aggregate=True, slice=slc, melted=True)
            # Spatial operations on rotated pole require the output crs be WGS84.
            self.assertEqual(ops.output_crs, CFWGS84())
            ret = ops.execute()
            if output_format == constants.OUTPUT_FORMAT_NUMPY:
                field = ret[1]['pr']
                self.assertEqual(field.shape, (1, 10, 1, 1, 1))
                value = ret.gvu(1, 'pr')
                self.assertAlmostEqual(manual_mean, value[0, 4, 0, 0, 0])
            else:
                with open(ret, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                self.assertEqual(len(rows), 10)
                self.assertAlmostEqual(float(rows[4]['VALUE']), manual_mean)

    @attr('data')
    def test_keyword_calc_grouping_none_date_parts(self):
        _cg = [None, ['day', 'month'], 'day']

        for cg in _cg:
            if cg is not None:
                eq = tuple(cg)
            else:
                eq = cg
            obj = definition.CalcGrouping(cg)
            try:
                self.assertEqual(obj.value, eq)
            except AssertionError:
                self.assertEqual(obj.value, ('day',))

        # # only month, year, and day combinations are currently supported
        rd = self.test_data.get_rd('cancm4_tas')
        calcs = [None, [{'func': 'mean', 'name': 'mean'}]]
        acceptable = ['day', 'month', 'year']
        for calc in calcs:
            for length in [1, 2, 3, 4, 5]:
                for combo in itertools.combinations(['day', 'month', 'year', 'hour', 'minute'], length):
                    try:
                        OcgOperations(dataset=rd, calc=calc, calc_grouping=combo)
                    except DefinitionValidationError:
                        reraise = True
                        for c in combo:
                            if c not in acceptable:
                                reraise = False
                        if reraise:
                            raise

    @attr('data')
    def test_keyword_calc_grouping_seasonal_with_unique(self):
        """Test calc_grouping argument using a seasonal unique flag."""

        calc_grouping = [[12, 1, 2], 'unique']
        calc = [{'func': 'mean', 'name': 'mean'}]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc_grouping=calc_grouping, geom='state_boundaries', select_ugid=[27],
                                  output_format='nc', calc=calc)
        ret = ops.execute()
        rd2 = ocgis.RequestDataset(uri=ret, variable='mean')
        field = rd2.get()
        self.assertIsNotNone(field.temporal.bounds)
        self.assertEqual(field.temporal.bounds_datetime.tolist(),
                         [[datetime.datetime(2001, 12, 1, 12, 0), datetime.datetime(2002, 2, 28, 12, 0)],
                          [datetime.datetime(2002, 12, 1, 12, 0), datetime.datetime(2003, 2, 28, 12, 0)],
                          [datetime.datetime(2003, 12, 1, 12, 0), datetime.datetime(2004, 2, 28, 12, 0)],
                          [datetime.datetime(2004, 12, 1, 12, 0), datetime.datetime(2005, 2, 28, 12, 0)],
                          [datetime.datetime(2005, 12, 1, 12, 0), datetime.datetime(2006, 2, 28, 12, 0)],
                          [datetime.datetime(2006, 12, 1, 12, 0), datetime.datetime(2007, 2, 28, 12, 0)],
                          [datetime.datetime(2007, 12, 1, 12, 0), datetime.datetime(2008, 2, 28, 12, 0)],
                          [datetime.datetime(2008, 12, 1, 12, 0), datetime.datetime(2009, 2, 28, 12, 0)],
                          [datetime.datetime(2009, 12, 1, 12, 0), datetime.datetime(2010, 2, 28, 12, 0)]])
        self.assertEqual(field.shape, (1, 9, 1, 3, 3))

    @attr('data')
    def test_keyword_calc_grouping_seasonal_with_year(self):
        calc_grouping = [[1, 2, 3], 'year']
        calc = [{'func': 'mean', 'name': 'mean'}]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            geom='state_boundaries', select_ugid=[25])
        ret = ops.execute()
        self.assertEqual(ret[25]['tas'].shape, (1, 10, 1, 5, 4))

    @attr('data')
    def test_keyword_calc_grouping_with_string_expression(self):
        """Test that no calculation grouping is allowed with a string expression."""

        calc = 'es=tas*3'
        calc_grouping = ['month']
        rd = self.test_data.get_rd('cancm4_tas')
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping)

    @attr('data')
    def test_keyword_callback(self):

        app = []

        def callback(perc, msg, append=app):
            append.append((perc, msg))

        # print(perc,msg)

        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tasmax_2011')
        dataset = [rd, rd2]
        for ds in dataset:
            ds.time_region = {'month': [6]}
        ops = ocgis.OcgOperations(dataset=dataset, geom='state_boundaries', select_ugid=[16, 17],
                                  calc_grouping=['month'],
                                  calc=[{'func': 'mean', 'name': 'mean'}, {'func': 'median', 'name': 'median'}],
                                  callback=callback)
        ops.execute()

        self.assertTrue(len(app) > 15)
        self.assertEqual(app[-1][0], 100.0)

    @attr('data')
    def test_keyword_conform_units_to(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd2.alias = 'foo'
        ops = OcgOperations(dataset=[rd1, rd2], conform_units_to='celsius')
        for ds in ops.dataset.itervalues():
            self.assertEqual(ds.conform_units_to, 'celsius')

        # Test conform argument is updated.
        ops.conform_units_to = 'fahrenheit'
        for ds in ops.dataset.itervalues():
            self.assertEqual(ds.conform_units_to, 'fahrenheit')

    @attr('data')
    def test_keyword_conform_units_to_bad_units(self):
        rd = self.test_data.get_rd('cancm4_tas')
        with self.assertRaises(RequestValidationError):
            OcgOperations(dataset=rd, conform_units_to='crap')

    @attr('esmf', 'data')
    def test_keyword_dataset_esmf(self):
        """Test with operations on an ESMF Field."""

        efield = self.get_esmf_field(with_bounds=True)
        output_format = OutputFormat.iter_possible()
        for kk in output_format:
            # Geojson may only be written with a WGS84 coordinate system.
            if kk == constants.OUTPUT_FORMAT_GEOJSON:
                output_crs = CFWGS84()
            else:
                output_crs = None
            try:
                ops = OcgOperations(dataset=efield, output_format=kk, prefix=kk, output_crs=output_crs)
            except DefinitionValidationError:
                self.assertEqual(kk, constants.OUTPUT_FORMAT_METADATA_JSON)
                continue
            ret = ops.execute()
            self.assertIsNotNone(ret)
        efield.destroy()

    @attr('data')
    def test_keyword_geom(self):
        geom = make_poly((37.762, 38.222), (-102.281, -101.754))
        g = definition.Geom(geom)
        self.assertEqual(type(g.value), tuple)
        self.assertEqual(g.value[0].single.geom.bounds, (-102.281, 37.762, -101.754, 38.222))

        g = definition.Geom(None)
        self.assertEqual(g.value, None)
        self.assertEqual(str(g), 'geom=None')

        g = definition.Geom('mi_watersheds')
        self.assertEqual(str(g), 'geom="mi_watersheds"')

        geoms = GeomCabinetIterator('mi_watersheds')
        g = definition.Geom(geoms)
        self.assertEqual(len(list(g.value)), 60)
        self.assertEqual(g._shp_key, 'mi_watersheds')

    @attr('data')
    def test_keyword_geom_having_changed_select_ugid(self):
        ops = OcgOperations(dataset=self.test_data.get_rd('cancm4_tas'),
                            geom='state_boundaries')
        self.assertEqual(len(list(ops.geom)), 51)
        ops.geom_select_uid = [16, 17]
        self.assertEqual(len(list(ops.geom)), 2)

    @attr('data')
    def test_keyword_geom_string(self):
        ops = OcgOperations(dataset=self.datasets, geom='state_boundaries')
        self.assertEqual(len(list(ops.geom)), 51)
        ops.geom = None
        self.assertEqual(ops.geom, None)
        ops.geom = 'mi_watersheds'
        self.assertEqual(len(list(ops.geom)), 60)
        ops.geom = [-120, 40, -110, 50]
        self.assertEqual(ops.geom[0].single.geom.bounds, (-120.0, 40.0, -110.0, 50.0))

    @attr('data')
    def test_keyword_headers(self):
        headers = ['did', 'value']
        for htype in [list, tuple]:
            hvalue = htype(headers)
            hh = definition.Headers(hvalue)
            self.assertEqual(hh.value, tuple(constants.HEADERS_REQUIRED + ['value']))

        headers = ['foo']
        with self.assertRaises(DefinitionValidationError):
            definition.Headers(headers)

        headers = []
        hh = definition.Headers(headers)
        self.assertEqual(hh.value, tuple(constants.HEADERS_REQUIRED))

    @attr('data')
    def test_keyword_level_range(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd.alias = 'foo'
        lr = [1, 2]
        ops = ocgis.OcgOperations(dataset=[rd, rd2], level_range=lr)
        for r in [rd, rd2]:
            self.assertEqual(r.level_range, None)
        for r in ops.dataset.itervalues():
            self.assertEqual(r.level_range, tuple(lr))

        lr = [2, 3]
        ops.level_range = lr
        for r in ops.dataset.itervalues():
            self.assertEqual(r.level_range, tuple(lr))

    @attr('data')
    def test_keyword_prefix(self):
        # the meta output format should not create an output directory
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, output_format=constants.OUTPUT_FORMAT_METADATA_OCGIS)
        ops.execute()
        self.assertEqual(len(os.listdir(self.current_dir_output)), 0)

    @attr('esmf', 'data', 'slow')
    def test_keyword_output_format_esmpy(self):
        """Test with the ESMPy output format."""
        import ESMF

        slc = [None, None, None, [0, 10], [0, 10]]
        kwds = dict(as_field=[False, True],
                    with_slice=[True, False])
        for k in self.iter_product_keywords(kwds):
            rd = self.test_data.get_rd('cancm4_tas')
            if k.as_field:
                rd = rd.get()
            if k.with_slice:
                slc = slc
            else:
                slc = None
            ops = OcgOperations(dataset=rd, output_format='esmpy', slice=slc)
            ret = ops.execute()
            self.assertIsInstance(ret, ESMF.Field)
            try:
                self.assertEqual(ret.data.shape, (1, 3650, 1, 10, 10))
            except AssertionError:
                self.assertFalse(k.with_slice)
                self.assertEqual(ret.data.shape, (1, 3650, 1, 64, 128))

    @attr('data')
    def test_keyword_output_format_nc_package_validation_raised_first(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('rotated_pole_ichec', kwds={'alias': 'tas2'})
        try:
            ocgis.OcgOperations(dataset=[rd, rd2], output_format=constants.OUTPUT_FORMAT_NETCDF)
        except DefinitionValidationError as e:
            self.assertIn('Data packages (i.e. more than one RequestDataset) may not be written to netCDF.',
                          e.message)

    @attr('data')
    def test_keyword_output_format_nc_2d_flexible_mesh_ugrid(self):
        rd = self.test_data.get_rd('cancm4_tas')
        output = constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH
        ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[25], output_format=output)
        ret = ops.execute()
        with self.nc_scope(ret) as ds:
            self.assertEqual(len(ds.dimensions['nMesh2_face']), 13)

    @attr('data')
    def test_keyword_output_format_options(self):
        # Test for netCDF output, unlimited dimensions are converted to fixed size.
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertTrue(field.temporal.unlimited)
        for b in [False, True]:
            output_format_options = {'unlimited_to_fixedsize': b}
            ops = OcgOperations(dataset=rd, snippet=True, output_format=constants.OUTPUT_FORMAT_NETCDF,
                                output_format_options=output_format_options, prefix=str(b))
            self.assertDictEqual(output_format_options, ops.output_format_options)
            ret = ops.execute()
            ocgis_unlimited = RequestDataset(ret).get().temporal.unlimited
            with self.nc_scope(ret) as ds:
                d = ds.dimensions[field.temporal.name]
                if b:
                    self.assertFalse(d.isunlimited())
                    self.assertFalse(ocgis_unlimited)
                else:
                    self.assertTrue(d.isunlimited())
                    self.assertTrue(ocgis_unlimited)

    @attr('data')
    def test_keyword_regrid_destination(self):
        """Test regridding not allowed with clip operation."""

        rd = self.test_data.get_rd('cancm4_tas')
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, regrid_destination=rd, spatial_operation='clip')

    @attr('data', 'esmf')
    def test_keyword_regrid_destination_to_nc(self):
        """Write regridded data to netCDF."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')

        ops = OcgOperations(dataset=rd1, regrid_destination=rd2, output_format='nc', snippet=True,
                            geom='state_boundaries', select_ugid=[25])
        ret = ops.execute()

        field = ocgis.RequestDataset(ret).get()
        self.assertIsNotNone(field.spatial.grid.corners)
        self.assertTrue(np.any(field.variables.first().value.mask))

    @attr('data', 'esmf')
    def test_keyword_regrid_destination_to_shp_vector_wrap(self):
        """Test writing to shapefile with different vector wrap options."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')

        for vector_wrap in [True, False]:
            ops = OcgOperations(dataset=rd1, regrid_destination=rd2, output_format='shp', snippet=True,
                                geom='state_boundaries', select_ugid=[25], vector_wrap=vector_wrap,
                                prefix=str(vector_wrap), melted=True)
            ret = ops.execute()
            sci = GeomCabinetIterator(path=ret)
            geoms = [element['geom'] for element in sci]
            for geom in geoms:
                if vector_wrap:
                    self.assertLess(geom.bounds[0], 0)
                else:
                    self.assertGreater(geom.bounds[0], 0)

    @attr('data')
    def test_keyword_spatial_operation(self):
        values = (None, 'clip', 'intersects')
        ast = ('intersects', 'clip', 'intersects')

        klass = definition.SpatialOperation
        for v, a in zip(values, ast):
            obj = klass(v)
            self.assertEqual(obj.value, a)

    @attr('data')
    def test_keyword_spatial_operations_bounding_box(self):
        geom = [-80, 22.5, 50, 70.0]
        rd = self.test_data.get_rd('subset_test_slp')
        ops = OcgOperations(dataset=rd, geom=geom)
        ret = ops.execute()
        field = ret[1]['slp']
        self.assertEqual(field.shape, (1, 365, 1, 18, 143))

    @attr('data')
    def test_keyword_spatial_reorder(self):
        rd = self.test_data.get_rd('cancm4_tas')

        field_2d = rd.get()[0, 0, 0, :, :]
        field_2d.spatial.grid.value
        field_2d.spatial.grid.corners
        field_2d.variables['tas'].value
        field_2d.spatial.grid.row = None
        field_2d.spatial.grid.col = None
        self.assertIsNone(field_2d.spatial.grid.col)

        kwds = {'dataset': [rd, field_2d], 'geom': [None, [-20, -20, 20, 20]]}

        for ctr, k in enumerate(self.iter_product_keywords(kwds)):
            # if ctr != 1: continue
            # print ctr, k
            original_value = rd.get()[:, 0, :, :, :].variables['tas'].value
            ops = OcgOperations(dataset=k.dataset, snippet=True, geom=k.geom, spatial_wrapping='wrap',
                                spatial_reorder=True)
            ret = ops.execute()
            field = ret[1]['tas']
            col_value = field.spatial.grid.value[1]
            actual_longitude = col_value[:, 0].mean()
            if k.geom is None:
                self.assertLess(actual_longitude, -170)
            else:
                # Test the subset is applied with reordering.
                self.assertGreater(actual_longitude, -30)
                self.assertLess(actual_longitude, 0)
            # Test the value arrays are not the same following a reorder.
            with self.assertRaises(AssertionError):
                self.assertNumpyAll(field.variables['tas'].value, original_value)

    @attr('data')
    def test_keyword_time_range(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd.alias = 'foo'
        tr = [datetime.datetime(2002, 1, 1), datetime.datetime(2002, 3, 1)]
        ops = ocgis.OcgOperations(dataset=[rd, rd2], time_range=tr)
        for r in [rd, rd2]:
            self.assertEqual(r.time_range, None)
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_range, tuple(tr))

        tr = [datetime.datetime(2002, 1, 1), datetime.datetime(2003, 3, 1)]
        ops.time_range = tr
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_range, tuple(tr))

    @attr('data')
    def test_keyword_time_range_and_time_region_null_parms(self):
        ops = OcgOperations(dataset=self.datasets_no_range)
        self.assertEqual(ops.geom, None)
        self.assertEqual(len(ops.dataset), 3)
        for ds in ops.dataset.itervalues():
            self.assertEqual(ds.time_range, None)
            self.assertEqual(ds.level_range, None)
        ops.__repr__()

    @attr('data')
    def test_keyword_time_region(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd.alias = 'foo'
        tr = {'month': [6], 'year': [2005]}
        ops = ocgis.OcgOperations(dataset=[rd, rd2], time_region=tr)
        for r in [rd, rd2]:
            self.assertEqual(r.time_region, None)
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_region, tr)

        tr = {'month': [6], 'year': [2006]}
        ops.time_region = tr
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_region, tr)

    @attr('data')
    def test_keyword_time_subset_func(self):

        def _func_(value, bounds=None):
            indices = []
            for ii, v in enumerate(value.flat):
                if v.month == 6:
                    indices.append(ii)
            return indices

        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, time_subset_func=_func_, geom='state_boundaries', geom_select_uid=[20])
        ret = ops.execute()
        for v in ret[20]['tas'].temporal.value_datetime:
            self.assertEqual(v.month, 6)

        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, time_subset_func=_func_, geom='state_boundaries', geom_select_uid=[20],
                            output_format=constants.OUTPUT_FORMAT_NETCDF)
        ret = ops.execute()
        rd_out = RequestDataset(ret)
        for v in rd_out.get().temporal.value_datetime:
            self.assertEqual(v.month, 6)

    @attr('data')
    def test_update_dependents(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, time_subset_func=lambda x, y: [1, 2])
        self.assertIsInstance(ops.dataset.first().time_subset_func, FunctionType)

    @attr('data')
    def test_validate(self):
        # snippets should be allowed for field objects
        field = self.test_data.get_rd('cancm4_tas').get()
        ops = OcgOperations(dataset=field, snippet=True)
        self.assertTrue(ops.snippet)

        # test driver validation is called appropriately
        path = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(path)
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, output_format='csv')


class TestOcgOperationsNoData(TestBase):
    @staticmethod
    def get_wrap_field(crs=None, unwrapped=True):
        row = VectorDimension(value=[20, 40])
        if unwrapped:
            col_value = [1, 180, 270]
        else:
            col_value = [-170, 0, 170]
        col = VectorDimension(value=col_value)
        grid = SpatialGridDimension(row=row, col=col)
        spatial = SpatialDimension(grid=grid, crs=crs)
        value = np.random.rand(2, 3).reshape(1, 1, 1, 2, 3)
        var = Variable(name='foo', value=value)
        field = Field(spatial=spatial, variables=var)

        return field

    def test_system_mftime(self):
        """Test a multi-file dataset with varying units on the time variables."""

        paths = create_mftime_nc_files(self, with_all_cf=True)
        rd = RequestDataset(paths)
        field = rd.get()
        self.assertIsNone(field.temporal._value)

        desired = [0., 1., 2., 366., 367., 368.]
        actual = field.temporal.value.tolist()
        self.assertEqual(actual, desired)

        ops = OcgOperations(dataset=rd, output_format=constants.OUTPUT_FORMAT_NETCDF)
        ret = ops.execute()

        out_rd = RequestDataset(uri=ret)
        self.assertEqual(out_rd.get().temporal.value.tolist(), desired)

    def test_keyword_spatial_wrapping(self):
        keywords = {'spatial_wrapping': list(SpatialWrapping.iter_possible()),
                    'crs': [None, Spherical(), CoordinateReferenceSystem(epsg=2136)],
                    'unwrapped': [True, False]}
        for k in self.iter_product_keywords(keywords):
            # print(k)
            field = self.get_wrap_field(crs=k.crs, unwrapped=k.unwrapped)

            ops = OcgOperations(dataset=field, spatial_wrapping=k.spatial_wrapping)
            ret = ops.execute()
            actual_field = ret[1]['foo']
            actual = actual_field.spatial.wrapped_state

            if k.crs != Spherical():
                desired = None
            else:
                p = k.spatial_wrapping
                if p is None:
                    if k.unwrapped:
                        desired = WrappedState.UNWRAPPED
                    else:
                        desired = WrappedState.WRAPPED
                elif p == 'wrap':
                    desired = WrappedState.WRAPPED
                else:
                    desired = WrappedState.UNWRAPPED

            self.assertEqual(actual, desired)
