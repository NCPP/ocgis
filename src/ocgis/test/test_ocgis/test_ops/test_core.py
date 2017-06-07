import csv
import datetime
import itertools
import os
import sys
from datetime import datetime as dt
from unittest import SkipTest

import numpy as np
from shapely.geometry import Point, LineString

import ocgis
from ocgis import RequestDataset, vm
from ocgis import constants
from ocgis import env
from ocgis.collection.field import Field
from ocgis.constants import WrappedState, HeaderName, OutputFormatName, DMK
from ocgis.exc import DefinitionValidationError
from ocgis.ops.core import OcgOperations
from ocgis.ops.parms import definition
from ocgis.ops.parms.definition import RegridOptions, OutputFormat, SpatialWrapping
from ocgis.spatial.geom_cabinet import GeomCabinetIterator, GeomCabinet
from ocgis.spatial.grid import Grid
from ocgis.test.base import TestBase, attr, create_gridxy_global, create_exact_field
from ocgis.test.test_simple.test_dependencies import create_mftime_nc_files
from ocgis.util.addict import Dict
from ocgis.util.helpers import make_poly, create_exact_field_value
from ocgis.variable.base import Variable
from ocgis.variable.crs import Spherical, CoordinateReferenceSystem, WGS84
from ocgis.variable.temporal import TemporalVariable
from ocgis.vmachine.mpi import OcgDist, MPI_RANK, variable_collection_scatter, MPI_COMM, dgather, \
    hgather, MPI_SIZE


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

        s = "STATE_NAME in ('Wisconsin', 'Vermont')"
        ops = OcgOperations(dataset=self.datasets, geom_select_sql_where=s, geom='state_boundaries')
        self.assertEqual(ops.geom_select_sql_where, s)
        self.assertEqual(len(ops.geom), 2)

    @attr('data')
    def test_str(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        ret = str(ops)
        self.assertTrue(str(ret).startswith('OcgOperations'))
        self.assertGreater(len(ret), 500)

    @attr('data')
    def test_system_scalar_level_dimension(self):
        """Test scalar level dimensions are not dropped in netCDF output."""

        rd = self.test_data.get_rd('cancm4_tas')
        desired_height_metadata = rd.metadata['variables']['height']
        ops = OcgOperations(dataset=rd, output_format='nc', snippet=True)
        ret = ops.execute()

        rd_out = RequestDataset(uri=ret)
        actual = rd_out.metadata['variables']['height']

        # Not worried about order of attributes.
        desired_height_metadata['attrs'] = dict(desired_height_metadata['attrs'])
        actual['attrs'] = dict(actual['attrs'])

        self.assertDictEqual(actual, desired_height_metadata)

    @attr('data')
    def test_get_base_request_size(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        size = ops.get_base_request_size()
        self.assertAlmostEqual(size['total'], 116890.046875)
        self.assertAsSetEqual(list(size['field']['tas'].keys()), list(rd.get().keys()))

        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, regrid_destination=rd).get_base_request_size()

    @attr('data')
    def test_get_base_request_size_multifile(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        rds = [rd1, rd2]
        ops = OcgOperations(dataset=rds)
        size = ops.get_base_request_size()
        actual = size['total']
        desired = 1784714.6640625
        self.assertAlmostEqual(actual, desired)

    @attr('data')
    def test_get_base_request_size_multifile_with_geom(self):
        rd1 = self.test_data.get_rd('cancm4_tas')

        rd2 = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        bdm = rd2.dimension_map
        bdm.set_variable(DMK.X, 'xc')
        bdm.set_variable(DMK.Y, 'yc')
        rd2 = self.test_data.get_rd('narccap_pr_wrfg_ncep', kwds={'dimension_map': bdm})

        rds = [rd1, rd2]
        ops = OcgOperations(dataset=rds, geom='state_boundaries', select_ugid=[23])
        size = ops.get_base_request_size()
        actual = size['total']
        desired = 22243.46484375
        self.assertAlmostEqual(actual, desired)

    @attr('data')
    def test_get_base_request_size_test_data(self):
        for key in list(self.test_data.keys()):
            rd = self.test_data.get_rd(key)
            ops = OcgOperations(dataset=rd)
            ret = ops.get_base_request_size()
            self.assertTrue(ret['total'] > 1)

    @attr('data')
    def test_get_base_request_size_with_calculation(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}],
                            calc_grouping=['month'])
        size = ops.get_base_request_size()
        self.assertEqual(size['field']['tas']['time']['shape'][0], 3650)

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
        self.assertEqual(k.value, 'auto')
        self.assertEqual(str(k), 'abstraction="auto"')

        k = kk('point')
        self.assertEqual(k.value, 'point')

        with self.assertRaises(DefinitionValidationError):
            kk('pt')

    @attr('data')
    def test_keyword_aggregate(self):
        rd = self.test_data.get_rd('rotated_pole_cnrm_cerfacs')

        ofield = rd.get().get_field_slice({'time': slice(0, 10), 'y': slice(0, 10), 'x': slice(0, 10)})
        ovalue = ofield['pr'].get_value()
        manual_mean = ovalue[4, :, :].mean()

        slc = [None, [0, 10], None, [0, 10], [0, 10]]
        for output_format in [OutputFormatName.OCGIS, OutputFormatName.CSV]:
            ops = OcgOperations(dataset=rd, output_format=output_format, aggregate=True, slice=slc, melted=True)
            # Spatial operations on rotated pole require the output be spherical.
            self.assertEqual(ops.output_crs, Spherical())
            ret = ops.execute()
            if output_format == constants.OutputFormatName.OCGIS:
                field = ret.get_element()
                self.assertEqual(field['pr'].shape, (10, 1))
                self.assertAlmostEqual(float(manual_mean), float(field['pr'].get_value()[4]))
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
        self.assertEqual(field.temporal.bounds.value_datetime.tolist(),
                         [[datetime.datetime(2001, 12, 1, 12, 0), datetime.datetime(2002, 2, 28, 12, 0)],
                          [datetime.datetime(2002, 12, 1, 12, 0), datetime.datetime(2003, 2, 28, 12, 0)],
                          [datetime.datetime(2003, 12, 1, 12, 0), datetime.datetime(2004, 2, 28, 12, 0)],
                          [datetime.datetime(2004, 12, 1, 12, 0), datetime.datetime(2005, 2, 28, 12, 0)],
                          [datetime.datetime(2005, 12, 1, 12, 0), datetime.datetime(2006, 2, 28, 12, 0)],
                          [datetime.datetime(2006, 12, 1, 12, 0), datetime.datetime(2007, 2, 28, 12, 0)],
                          [datetime.datetime(2007, 12, 1, 12, 0), datetime.datetime(2008, 2, 28, 12, 0)],
                          [datetime.datetime(2008, 12, 1, 12, 0), datetime.datetime(2009, 2, 28, 12, 0)],
                          [datetime.datetime(2009, 12, 1, 12, 0), datetime.datetime(2010, 2, 28, 12, 0)]])
        self.assertEqual(field['mean'].shape, (9, 3, 3))

    @attr('data')
    def test_keyword_calc_grouping_seasonal_with_year(self):
        calc_grouping = [[1, 2, 3], 'year']
        calc = [{'func': 'mean', 'name': 'mean'}]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            geom='state_boundaries', select_ugid=[25])
        ret = ops.execute()
        self.assertEqual(ret.get_element(variable_name='mean').shape, (10, 4, 4))

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

    @attr('data', 'cfunits')
    def test_keyword_conform_units_to(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd2._field_name = 'foo'
        ops = OcgOperations(dataset=[rd1, rd2], conform_units_to='celsius', snippet=True)
        ret = ops.execute()

        original = rd1.get().get_field_slice({'time': 0})
        original = original.data_variables[0].get_value().sum()

        for field in ret.iter_fields():
            actual_sum = field.data_variables[0].get_value().sum()
            diff = actual_sum - original
            self.assertFalse(np.isclose(diff, 0))

    @attr('data', 'cfunits')
    def test_keyword_conform_units_to_bad_units(self):
        rd = self.test_data.get_rd('cancm4_tas')
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, conform_units_to='crap')

    @attr('esmf', 'data')
    def test_keyword_dataset_esmf(self):
        """Test with operations on an ESMF Field."""

        from ocgis.regrid.base import get_esmf_field_from_ocgis_field

        efield = get_esmf_field_from_ocgis_field(self.get_field(nrlz=0, nlevel=0))
        output_format = OutputFormat.iter_possible()
        for kk in output_format:
            # Geojson may only be written with a spherical coordinate system.
            if kk == constants.OutputFormatName.GEOJSON:
                output_crs = WGS84()
            else:
                output_crs = None
            try:
                ops = OcgOperations(dataset=efield, output_format=kk, prefix=kk, output_crs=output_crs)
            except DefinitionValidationError:
                self.assertEqual(kk, constants.OutputFormatName.METADATA_JSON)
                continue
            ret = ops.execute()
            self.assertIsNotNone(ret)
        efield.destroy()

    @attr('data')
    def test_keyword_geom(self):
        geom = make_poly((37.762, 38.222), (-102.281, -101.754))
        g = definition.Geom(geom)
        self.assertEqual(type(g.value), tuple)
        self.assertEqual(g.value[0].geom.get_value()[0].bounds, (-102.281, 37.762, -101.754, 38.222))

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
        self.assertEqual(ops.geom[0].geom.get_value()[0].bounds, (-120.0, 40.0, -110.0, 50.0))

    @attr('data')
    def test_keyword_level_range(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd._field_name = 'foo'
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd2._field_name = 'foo2'
        lr = [1, 2]
        ops = ocgis.OcgOperations(dataset=[rd, rd2], level_range=lr)
        ret = ops.execute()
        for n in ['foo', 'foo2']:
            actual = ret.get_element(field_name=n)
            shp = actual.data_variables[0].shape
            self.assertEqual((3650, 64, 128), shp)
        for r in [rd, rd2]:
            self.assertEqual(r.level_range, None)

    @attr('data')
    def test_keyword_prefix(self):
        # the meta output format should not create an output directory
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, output_format=constants.OutputFormatName.METADATA_OCGIS)
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
                self.assertEqual(ret.data.shape, (3650, 10, 10))
            except AssertionError:
                self.assertFalse(k.with_slice)
                self.assertEqual(ret.data.shape, (3650, 64, 128))

    @attr('data')
    def test_keyword_output_format_nc_package_validation_raised_first(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('rotated_pole_ichec', kwds={'rename_variable': 'tas2'})
        try:
            ocgis.OcgOperations(dataset=[rd, rd2], output_format=constants.OutputFormatName.NETCDF)
        except DefinitionValidationError as e:
            self.assertIn('Data packages (i.e. more than one RequestDataset) may not be written to netCDF.',
                          e.message)

    @attr('data')
    def test_keyword_output_format_options(self):
        # Test for netCDF output, unlimited dimensions are converted to fixed size.
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertTrue(field.temporal.dimensions[0].is_unlimited)
        for b in [False, True]:
            output_format_options = {'unlimited_to_fixedsize': b}
            ops = OcgOperations(dataset=rd, snippet=True, output_format=constants.OutputFormatName.NETCDF,
                                output_format_options=output_format_options, prefix=str(b))
            self.assertDictEqual(output_format_options, ops.output_format_options)
            ret = ops.execute()
            ocgis_unlimited = RequestDataset(ret).get().temporal.dimensions[0].is_unlimited
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
        self.assertTrue(field.grid.has_bounds)
        self.assertTrue(np.any(field.data_variables[0].get_mask()))

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
        field = ret.get_element()
        self.assertEqual(field.data_variables[0].shape, (365, 20, 144))

    @attr('data')
    def test_keyword_spatial_reorder(self):
        rd = self.test_data.get_rd('cancm4_tas')

        kwds = {'dataset': [rd],
                'geom': [None, [-20, -20, 20, 20]]}

        for ctr, k in enumerate(self.iter_product_keywords(kwds)):
            original_value = rd.get().get_field_slice({'time': 0})['tas'].get_value()
            ops = OcgOperations(dataset=k.dataset, snippet=True, geom=k.geom, spatial_wrapping='wrap',
                                spatial_reorder=True)
            ret = ops.execute()
            field = ret.get_element()
            col_value = field.grid.x.get_value()
            actual_longitude = col_value[0:3].mean()
            if k.geom is None:
                self.assertLess(actual_longitude, -170)
            else:
                # Test the subset is applied with reordering.
                self.assertGreater(actual_longitude, -30)
                self.assertLess(actual_longitude, 0)
            # Test the value arrays are not the same following a reorder.
            with self.assertRaises(AssertionError):
                self.assertNumpyAll(field['tas'].get_value(), original_value)

    @attr('data')
    def test_keyword_time_range(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd._field_name = 'f1'
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd2._field_name = 'f2'

        self.assertEqual(rd.get().time.calendar, '365_day')

        tr = [datetime.datetime(2002, 1, 1), datetime.datetime(2002, 3, 1)]
        ops = ocgis.OcgOperations(dataset=[rd, rd2], time_range=tr)
        ret = ops.execute()
        for field in ret.iter_fields():
            self.assertEqual(field.time.extent, (55480.0, 55540.0))

    @attr('data')
    def test_keyword_time_range_and_time_region_null_parms(self):
        ops = OcgOperations(dataset=self.datasets_no_range)
        self.assertEqual(ops.geom, None)
        self.assertEqual(len(list(ops.dataset)), 3)
        for ds in ops.dataset:
            self.assertEqual(ds.time_range, None)
            self.assertEqual(ds.level_range, None)
        ops.__repr__()

    @attr('data')
    def test_keyword_time_region(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd._field_name = 'f1'
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd2._field_name = 'f2'

        tr = {'month': [6], 'year': [2005]}
        ops = ocgis.OcgOperations(dataset=[rd, rd2], time_region=tr)
        ret = ops.execute()

        for field in ret.iter_fields():
            self.assertEqual(field.time.extent, (56726.0, 56756.0))

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
        ret = ret.get_element()
        for v in ret.temporal.value_datetime:
            self.assertEqual(v.month, 6)

        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, time_subset_func=_func_, geom='state_boundaries', geom_select_uid=[20],
                            output_format=constants.OutputFormatName.NETCDF)
        ret = ops.execute()
        rd_out = RequestDataset(ret)
        for v in rd_out.get().temporal.value_datetime:
            self.assertEqual(v.month, 6)

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
        ompi = OcgDist()
        ompi.create_dimension('x', 5, dist=False)
        ompi.create_dimension('y', 7, dist=True)
        ompi.create_dimension('time', size_current=4, dist=False)
        ompi.update_dimension_bounds()

        if MPI_RANK == 0:
            row = Variable(value=[-60, -40, -20, 0, 20, 40, 60], name='y', dimensions='y')
            if unwrapped:
                col_value = [1, 90, 180, 225, 270]
            else:
                col_value = [-170, -85, 0, 85, 170]
            col = Variable(value=col_value, name='x', dimensions='x')
            grid = Grid(col, row)
            value = np.zeros((4, 7, 5))
            for col_idx in range(value.shape[-1]):
                value[:, :, col_idx] = col_idx
            time = TemporalVariable(name='time', value=[1, 2, 3, 4], dimensions='time')
            var = Variable(name='foo', value=value, dimensions=['time', 'y', 'x'])
            field = Field(grid=grid, is_data=var, crs=crs, time=time)
        else:
            field = None
        field = variable_collection_scatter(field, ompi)

        return field

    def test_system_dataset_identifiers_on_variables(self):
        """Test dataset identifiers make it to output variables for iteration."""

        paths = []
        variables = []
        for suffix in [1, 2]:
            path = self.get_temporary_file_path('foo{}.nc'.format(suffix))
            paths.append(path)
            x = Variable(name='x{}'.format(suffix), value=[2, 3], dimensions='x')
            y = Variable(name='y{}'.format(suffix), value=[4, 5, 6], dimensions='y')
            data_variable_name = 'data{}'.format(suffix)
            variables.append(data_variable_name)
            data = Variable(name=data_variable_name, value=np.arange(6).reshape(2, 3) + suffix,
                            dimensions=['x', 'y'])
            grid = Grid(x, y)
            field = Field(grid=grid, is_data=data)
            field.write(path)

        rds = [RequestDataset(uri=p, variable=dv) for p, dv in zip(paths, variables)]
        ops = OcgOperations(dataset=rds)
        rds_uids = [ds.uid for ds in ops.dataset]
        self.assertEqual(rds_uids, [1, 2])
        ret = ops.execute()

        for field in ret.iter_fields():
            self.assertFalse(field.grid.has_allocated_abstraction_geometry)
            for variable in list(field.values()):
                if isinstance(variable, CoordinateReferenceSystem):
                    continue
                self.assertIsNotNone(variable._request_dataset.uid)
                for row in variable.get_iter():
                    self.assertIsNotNone(row[HeaderName.DATASET_IDENTIFER])

    def test_system_field_is_untouched(self):
        """Test field is untouched if passed through operations with nothing happening."""

        field = self.get_field()
        gid_name = HeaderName.ID_GEOMETRY
        self.assertNotIn(gid_name, field)
        ops = OcgOperations(dataset=field, output_format=constants.OutputFormatName.OCGIS)
        ret = ops.execute()
        actual = ret.get_element()
        self.assertEqual(list(field.keys()), list(actual.keys()))

    def test_system_file_only(self):
        """Test file only writing."""

        field = self.get_field(nlevel=3, nrlz=4, ntime=62)

        for fo in [True, False]:
            ops = OcgOperations(dataset=field, calc=[{'func': 'mean', 'name': 'mean'}], calc_grouping=['month'],
                                output_format='nc', file_only=fo, prefix=str(fo))
            ret = ops.execute()

            out_field = RequestDataset(ret).get()
            for var in list(out_field.values()):
                if fo and var.name == 'mean':
                    self.assertFalse(var.has_allocated_value)
                else:
                    self.assertIsNone(var.get_mask())

            actual = out_field['mean']
            if fo:
                self.assertTrue(actual.get_mask().all())
            else:
                self.assertTrue(actual.get_value().sum() > 5)

    def test_system_geometry_identifer_added(self):
        """Test geometry identifier is added for linked dataset geometry formats."""

        field = self.get_field()
        gid_name = HeaderName.ID_GEOMETRY
        self.assertNotIn(gid_name, field)
        ops = OcgOperations(dataset=field, output_format=constants.OutputFormatName.CSV_SHAPEFILE)
        ret = ops.execute()

        csv_field = RequestDataset(ret).get()
        self.assertIn(gid_name, list(csv_field.keys()))
        shp_path = os.path.join(ops.dir_output, ops.prefix, 'shp', ops.prefix + '_gid.shp')
        shp_field = RequestDataset(shp_path).get()
        self.assertIn(gid_name, list(shp_field.keys()))

    def test_system_line_subsetting(self):
        """Test subsetting with a line."""

        line = LineString([(-0.4, 0.2), (1.35, 0.3), (1.38, -0.716)])
        geom = [{'geom': line, 'crs': None}]

        x = Variable('x', [-1, -0.5, 0.5, 1.5, 2], 'x')
        y = Variable('y', [-0.5, 0.5, 1.5], 'y')
        grid = Grid(x, y)
        grid.set_extrapolated_bounds('x_bounds', 'y_bounds', 'bounds')
        field = Field(grid=grid)

        ops = OcgOperations(dataset=field, geom=geom)
        ret = ops.execute()
        field = ret.get_element()

        desired = [[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], [[-0.5, 0.5, 1.5], [-0.5, 0.5, 1.5]]]
        actual = field.grid.get_value_stacked().tolist()
        self.assertEqual(actual, desired)

        desired = [[True, True, False], [False, False, False]]
        actual = field.grid.get_mask().tolist()
        self.assertEqual(actual, desired)

    def test_system_mftime(self):
        """Test a multi-file dataset with varying units on the time variables."""

        paths = create_mftime_nc_files(self, with_all_cf=True)
        rd = RequestDataset(paths)
        field = rd.get()
        self.assertIsNone(field.temporal._value)

        desired = [0., 1., 2., 366., 367., 368.]
        actual = field.temporal.get_value().tolist()
        self.assertEqual(actual, desired)

        ops = OcgOperations(dataset=rd, output_format=constants.OutputFormatName.NETCDF)
        ret = ops.execute()

        out_rd = RequestDataset(uri=ret)
        self.assertEqual(out_rd.get().temporal.get_value().tolist(), desired)

    def test_system_multiple_netcdf_files(self):
        """Test subsetting multiple netCDF files and returning a spatial collection."""

        grid = create_gridxy_global(resolution=3.0)
        vars = ['ocgis_example_tasmin', 'ocgis_example_tas', 'ocgis_example_tasmax']
        paths = [self.get_temporary_file_path('{}.nc'.format(ii)) for ii in vars]

        geom_select_uid = [16, 23]
        field_names = ['tasmin', 'tas', 'tasmax']
        for ctr, (path, var) in enumerate(zip(paths, vars), start=1):
            field = create_exact_field(grid.copy(), var, ntime=3)
            field.data_variables[0].get_value()[:] = 10 * ctr
            field.write(path)

        rds = [RequestDataset(uri=uri, variable=var, field_name=field_name) for uri, var, field_name in
               zip(paths, vars, field_names)]
        ops = OcgOperations(dataset=rds, spatial_operation='clip', aggregate=True, geom=self.path_state_boundaries,
                            geom_select_uid=geom_select_uid)
        ret = ops.execute()

        self.assertAsSetEqual(ret.children.keys(), geom_select_uid)
        for geom_uid in geom_select_uid:
            actual = ret.children[geom_uid].children.keys()
            self.assertAsSetEqual(actual, field_names)

            for idx, field_name in enumerate(field_names):
                actual = ret.get_element(container_ugid=geom_uid, field_name=field_names[idx], variable_name=vars[idx])
                actual = actual.get_value()
                actual = actual == (idx + 1) * 10
                self.assertTrue(np.all(actual))

    def test_system_netcdf_output_format(self):
        path = self.get_temporary_file_path('foo.nc')
        var = Variable('vec', value=[1, 2, 3, 4, 5], dimensions='dvec', dtype=np.int32)
        var.write(path)

        with self.nc_scope(path, 'r') as ds:
            self.assertEqual(ds.data_model, 'NETCDF4')

        rd = RequestDataset(uri=path)
        ops = OcgOperations(dataset=rd, prefix='converted', output_format='nc',
                            output_format_options={'data_model': 'NETCDF4_CLASSIC'})
        ret = ops.execute()

        with self.nc_scope(ret, 'r') as ds:
            self.assertEqual(ds.data_model, 'NETCDF4_CLASSIC')

    @attr('cfunits')
    def test_system_request_dataset_modifiers(self):
        """
        Test request dataset arguments are applied in operations to fields. There are parameters that may be passed to a
        request dataset or used solely in operations.
        """

        def _the_func_(arr, bounds=None):
            tfret = []
            for ctr, element in enumerate(arr.flat):
                if element.year == 2004:
                    tfret.append(ctr)
            return tfret

        time = TemporalVariable(name='my_time', value=[400, 800, 1200, 1600, 2000, 2400], dimensions='time_dimension',
                                units='days since 2001-1-1')
        level = Variable(name='my_level', value=[20, 30, 40, 50], dimensions='level_dimension')
        np.random.seed(1)
        original_value = np.random.rand(time.shape[0], level.shape[0])
        data = Variable(name='data',
                        value=original_value,
                        dimensions=['time_dimension', 'level_dimension'],
                        units='fahrenheit')
        field = Field(time=time, level=level, is_data=data)

        ops = OcgOperations(dataset=field,
                            time_range=[datetime.datetime(2003, 1, 1), datetime.datetime(2007, 1, 1)],
                            time_region={'year': [2003, 2004, 2005]},
                            time_subset_func=_the_func_,
                            level_range=[30, 40],
                            conform_units_to='celsius')
        ret = ops.execute()
        actual = ret.get_element()
        self.assertEqual(actual.time.shape, (1,))
        self.assertEqual(actual.time.value_datetime[0].year, 2004)
        self.assertEqual(actual.level.get_value().tolist(), [30, 40])
        self.assertAlmostEqual(actual.data_variables[0].get_value().mean(), -17.511663542109229)
        self.assertEqual(actual.data_variables[0].units, 'celsius')

    def test_system_user_geometry_identifer_added(self):
        """Test geometry identifier is added for linked dataset geometry formats."""

        field = self.get_field(crs=WGS84())
        subset_geom = Point(field.grid.x.get_value()[0], field.grid.y.get_value()[0]).buffer(0.1)
        ops = OcgOperations(dataset=field, geom=subset_geom, output_format=constants.OutputFormatName.CSV_SHAPEFILE)
        ret = ops.execute()

        ugid_name = HeaderName.ID_SELECTION_GEOMETRY
        csv_field = RequestDataset(ret).get()

        self.assertIn(ugid_name, list(csv_field.keys()))
        shp_path = os.path.join(ops.dir_output, ops.prefix, 'shp', ops.prefix + '_gid.shp')
        shp_field = RequestDataset(shp_path).get()
        self.assertIn(ugid_name, list(shp_field.keys()))

        shp_path = os.path.join(ops.dir_output, ops.prefix, 'shp', ops.prefix + '_ugid.shp')
        shp_field = RequestDataset(shp_path).get()
        self.assertIn(ugid_name, list(shp_field.keys()))

    @attr('mpi')
    def test_system_spatial_averaging_through_operations(self):
        data_name = 'data'

        with vm.scoped('write', [0]):
            if not vm.is_null:
                x = Variable('x', range(5), 'x', float)
                y = Variable('y', range(7), 'y', float)
                grid = Grid(x, y)

                data_value = np.arange(x.size * y.size).reshape(grid.shape)
                data = Variable(data_name, data_value, grid.dimensions, float)
                data_value = data.get_value()

                field = Field(grid=grid, is_data=data)

                path = self.get_temporary_file_path('data.nc')
                field.write(path)
            else:
                data_value, path = None, None
        data_value = MPI_COMM.bcast(data_value)
        path = MPI_COMM.bcast(path)

        rd = RequestDataset(path, variable=data_name)

        ops = OcgOperations(dataset=rd, aggregate=True)
        ret = ops.execute()
        if ret is None:
            self.assertNotEqual(vm.rank, vm.root)
        else:
            out_field = ret.get_element()

            if MPI_RANK == 0:
                desired = data_value.mean()
                actual = out_field.data_variables[0].get_value()[0]
                self.assertEqual(actual, desired)

    @attr('release')
    def test_system_spatial_averaging_through_operations_state_boundaries(self):
        if MPI_SIZE != 8:
            raise SkipTest('MPI_SIZE != 8')

        ntime = 3
        # Get the exact field value for the state's representative center.
        with vm.scoped([0]):
            if MPI_RANK == 0:
                states = RequestDataset(self.path_state_boundaries, driver='vector').get()
                states.update_crs(env.DEFAULT_COORDSYS)
                fill = np.zeros((states.geom.shape[0], 2))
                for idx, geom in enumerate(states.geom.get_value().flat):
                    centroid = geom.centroid
                    fill[idx, :] = centroid.x, centroid.y
                exact_states = create_exact_field_value(fill[:, 0], fill[:, 1])
                state_ugid = states['UGID'].get_value()
                area = states.geom.area

        keywords = {
            'spatial_operation': [
                'clip',
                'intersects'
            ],
            'aggregate': [
                True,
                False
            ],
            'wrapped': [True, False],
            'output_format': [
                OutputFormatName.OCGIS,
                'csv',
                'csv-shp',
                'shp'
            ],
        }

        # total_iterations = len(list(self.iter_product_keywords(keywords)))

        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            # barrier_print(k)
            # if ctr % 1 == 0:
            #     if vm.is_root:
            #         print('Iteration {} of {}...'.format(ctr + 1, total_iterations))

            with vm.scoped([0]):
                if vm.is_root:
                    grid = create_gridxy_global(resolution=1.0, dist=False, wrapped=k.wrapped)
                    field = create_exact_field(grid, 'foo', ntime=ntime)
                    path = self.get_temporary_file_path('foo.nc')
                    field.write(path)
                else:
                    path = None
            path = MPI_COMM.bcast(path)

            rd = RequestDataset(path)

            ops = OcgOperations(dataset=rd, geom='state_boundaries', spatial_operation=k.spatial_operation,
                                aggregate=k.aggregate, output_format=k.output_format, prefix=str(ctr),
                                # geom_select_uid=[8]
                                )
            ret = ops.execute()

            # Test area is preserved for a problem element during union. The union's geometry was not fully represented
            # in the output.
            if k.output_format == 'shp' and k.aggregate and k.spatial_operation == 'clip':
                with vm.scoped([0]):
                    if vm.is_root:
                        inn = RequestDataset(ret).get()
                        inn_ugid_idx = np.where(inn['UGID'].get_value() == 8)[0][0]
                        ugid_idx = np.where(state_ugid == 8)[0][0]
                        self.assertAlmostEqual(inn.geom.get_value()[inn_ugid_idx].area, area[ugid_idx], places=2)

            # Test the overview geometry shapefile is written.
            if k.output_format == 'shp':
                directory = os.path.split(ret)[0]
                contents = os.listdir(directory)
                actual = ['_ugid.shp' in c for c in contents]
                self.assertTrue(any(actual))
            elif k.output_format == 'csv-shp':
                directory = os.path.split(ret)[0]
                directory = os.path.join(directory, 'shp')
                contents = os.listdir(directory)
                actual = ['_ugid.shp' in c for c in contents]
                self.assertTrue(any(actual))
                if not k.aggregate:
                    actual = ['_gid.shp' in c for c in contents]
                    self.assertTrue(any(actual))

            if k.output_format == OutputFormatName.OCGIS:
                geom_keys = ret.children.keys()
                all_geom_keys = vm.gather(np.array(geom_keys))
                if vm.is_root:
                    all_geom_keys = hgather(all_geom_keys)
                    self.assertEqual(len(np.unique(all_geom_keys)), 51)

                if k.aggregate:
                    actual = Dict()
                    for field, container in ret.iter_fields(yield_container=True):
                        if not field.is_empty:
                            ugid = container.geom.ugid.get_value()[0]
                            actual[ugid]['actual'] = field.data_variables[0].get_value()
                            actual[ugid]['area'] = container.geom.area[0]

                    actual = vm.gather(actual)

                    if vm.is_root:
                        actual = dgather(actual)

                        ares = []
                        actual_areas = []
                        for ugid_key, v in actual.items():
                            ugid_idx = np.where(state_ugid == ugid_key)[0][0]
                            desired = exact_states[ugid_idx]
                            actual_areas.append(v['area'])
                            for tidx in range(ntime):
                                are = np.abs((desired + ((tidx + 1) * 10)) - v['actual'][tidx, 0])
                                ares.append(are)

                        if k.spatial_operation == 'clip':
                            diff = np.abs(np.array(area) - np.array(actual_areas))
                            self.assertLess(np.max(diff), 1e-6)
                            self.assertLess(np.mean(diff), 1e-6)

                        # Test relative errors.
                        self.assertLess(np.max(ares), 0.031)
                        self.assertLess(np.mean(ares), 0.009)

    @attr('mpi', 'no-3.5')
    def test_system_spatial_wrapping_and_reorder(self):
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            raise SkipTest('undefined behavior with Python 3.5')

        keywords = {'spatial_wrapping': list(SpatialWrapping.iter_possible()),
                    'crs': [None, Spherical(), CoordinateReferenceSystem(epsg=2136)],
                    'unwrapped': [True, False],
                    'spatial_reorder': [False, True]}
        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            field = self.get_wrap_field(crs=k.crs, unwrapped=k.unwrapped)

            ops = OcgOperations(dataset=field, spatial_wrapping=k.spatial_wrapping, spatial_reorder=k.spatial_reorder)
            ret = ops.execute()

            actual_field = ret.get_element()

            with vm.scoped_by_emptyable('wrapped state', actual_field):
                if not vm.is_null:
                    actual = actual_field.wrapped_state
                else:
                    actual = None
            actual_x = actual_field.grid.x.get_value()

            if not actual_field.is_empty:
                self.assertLessEqual(actual_x.max(), 360.)
                if k.spatial_reorder and k.unwrapped and k.spatial_wrapping == 'wrap' and k.crs == Spherical():
                    actual_data_value = actual_field.data_variables[0].get_value()
                    desired_reordered = [None] * actual_data_value.shape[1]
                    for idx in range(actual_data_value.shape[1]):
                        desired_reordered[idx] = [3.0, 4.0, 0.0, 1.0, 2.0]
                    for tidx in range(actual_data_value.shape[0]):
                        time_data_value = actual_data_value[tidx]
                        self.assertEqual(time_data_value.tolist(), desired_reordered)

                if k.spatial_reorder and not k.unwrapped and not k.spatial_wrapping:
                    self.assertTrue(actual_x[0] < actual_x[-1])

            if actual is None or k.crs != Spherical():
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
