import os
from copy import deepcopy

import numpy as np

import ocgis
from ocgis import constants
from ocgis import env
from ocgis.api.operations import OcgOperations
from ocgis.api.request.base import RequestDataset
from ocgis.conv.nc import NcConverter, NcUgrid2DFlexibleMeshConverter
from ocgis.exc import DefinitionValidationError
from ocgis.test.base import attr
from ocgis.test.base import nc_scope
from ocgis.test.test_ocgis.test_conv.test_base import AbstractTestConverter
from ocgis.util.geom_cabinet import GeomCabinet
from ocgis.util.itester import itr_products_keywords


class TestNcConverter(AbstractTestConverter):
    @attr('data')
    def test_fill_value_modified(self):
        # Test the fill value is appropriately copied if reset inside the field.
        coll = self.get_spatial_collection()
        ref = coll[1]['tas'].variables['tas']
        ref._dtype = np.int32
        ref._value = ref.value.astype(np.int32)
        ref._fill_value = None
        ncconv = NcConverter([coll], outdir=self.current_dir_output, prefix='ocgis_output')
        ret = ncconv.write()
        with nc_scope(ret) as ds:
            var = ds.variables['tas']
            self.assertEqual(var.dtype, np.dtype('int32'))
            self.assertEqual(var.shape, (1, 1, 1))
            self.assertEqual(var._FillValue, np.ma.array([], dtype=np.dtype('int32')).fill_value)

    @attr('data')
    def test_fill_value_copied(self):
        rd = self.test_data.get_rd('cancm4_tas')
        with nc_scope(rd.uri) as ds:
            fill_value_test = ds.variables['tas']._FillValue
        ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            self.assertEqual(fill_value_test, ds.variables['tas']._FillValue)

    @attr('data')
    def test_get_file_format(self):
        # Test the file format is pulled from the environment and not from constants.
        env.NETCDF_FILE_FORMAT = 'NETCDF3_CLASSIC'
        coll = self.get_spatial_collection(field=self.get_field())
        conv = NcConverter([coll], outdir=self.current_dir_output, prefix='foo')
        file_format = conv._get_file_format_()
        self.assertEqual(file_format, 'NETCDF3_CLASSIC')
        env.reset()

        # Pass a data model directly.
        options = {'data_model': 'foo'}
        conv = NcConverter([coll], outdir=self.current_dir_output, prefix='foo', options=options)
        self.assertEqual(conv._get_file_format_(), 'foo')

        # Use a field as the input dataset.
        coll = self.get_spatial_collection(field=self.get_field())
        conv = NcConverter([coll], outdir=self.current_dir_output, prefix='foo')
        file_format = conv._get_file_format_()
        self.assertEqual(file_format, env.NETCDF_FILE_FORMAT)

        # Add operations with a field as the dataset.
        ops = OcgOperations(dataset=coll[1]['foo'], output_format='nc')
        conv = NcConverter([coll], outdir=self.current_dir_output, prefix='foo', ops=ops)
        file_format = conv._get_file_format_()
        self.assertEqual(file_format, env.NETCDF_FILE_FORMAT)

        # Add operations and use a request dataset.
        coll = self.get_spatial_collection()
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, output_format='nc')
        conv = NcConverter([coll], outdir=self.current_dir_output, prefix='foo', ops=ops)
        file_format = conv._get_file_format_()
        with nc_scope(rd.uri) as ds:
            self.assertEqual(file_format, ds.file_format)

        # Use a shapefile as the input format.
        path = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=path)
        of = constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH
        ops = OcgOperations(dataset=rd, output_format=of)
        conv = NcConverter([coll], outdir=self.current_dir_output, prefix='foo', ops=ops)
        file_format = conv._get_file_format_()
        self.assertEqual(file_format, env.NETCDF_FILE_FORMAT)

    @attr('data')
    def test_write_coll(self):
        # Use a field as the input dataset.
        field = self.get_field()
        coll = self.get_spatial_collection(field=field)

        kwds = dict(with_ops=[False, True],
                    file_only=[False, True])

        for k in itr_products_keywords(kwds, as_namedtuple=True):

            if k.with_ops:
                ops = OcgOperations(dataset=self.test_data.get_rd('cancm4_tas'), file_only=k.file_only,
                                    output_format='nc', calc=[{'name': 'mean', 'func': 'mean'}],
                                    calc_grouping=['month'])
            else:
                ops = None

            conv = NcConverter([coll], outdir=self.current_dir_output, prefix='foo', ops=ops, overwrite=True)

            with nc_scope(conv.path, 'w') as ds:
                conv._write_coll_(ds, coll)
            with nc_scope(conv.path) as ds:
                value_nc = ds.variables['foo'][:]
                value_field = field.variables['foo'].value.squeeze()
                try:
                    self.assertNumpyAll(value_field, np.ma.array(value_nc))
                except AssertionError:
                    self.assertTrue(k.file_only)
                    self.assertTrue(k.with_ops)
                    self.assertTrue(value_nc.mask.all())
                self.assertIn('ocgis', ds.history)
                if k.with_ops:
                    self.assertIn('OcgOperations', ds.history)


class TestNcUgrid2DFlexibleMeshConverter(AbstractTestConverter):
    def test_init(self):
        self.assertEqual(NcUgrid2DFlexibleMeshConverter.__bases__, (NcConverter,))

    @attr('data')
    def test_validate_ops(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = deepcopy(rd1)
        rd2.alias = 'tas2'
        output_format = constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=[rd1, rd2], output_format=output_format)
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd1, output_format=output_format, file_only=True)
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd1, output_format=output_format, abstraction='point')

        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=uri, s_abstraction='point')
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, output_format=output_format)

        rd = RequestDataset(uri=uri, s_abstraction='point').get()
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, output_format=output_format)

    @attr('data')
    def test_write_archetype(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[25]).execute()
        field = coll[25]['tas']
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            NcUgrid2DFlexibleMeshConverter._write_archetype_(field, ds, None, {})
        with self.nc_scope(path) as ds:
            self.assertEqual(len(ds.dimensions['nMesh2_face']), 13)

        # Test with the polygons as none.
        field.spatial.geom._polygon = None
        field.spatial.geom.grid = None
        with self.assertRaises(ValueError):
            NcUgrid2DFlexibleMeshConverter._write_archetype_(field, None, None, {})

    def test_write_archetype_from_shapefile(self):
        """Test writing from a shapefile."""

        uri = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri)
        field = rd.get()
        sub = field[0, 0, 0, 0, [15, 33]]
        coll = sub.as_spatial_collection()
        conv = NcUgrid2DFlexibleMeshConverter([coll], outdir=self.current_dir_output, prefix='foo')
        ret = conv.write()
        with self.nc_scope(ret) as ds:
            self.assertEqual(len(ds.dimensions['nMesh2_face']), 2)

        of = constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH
        for a in [True, False]:
            ops = OcgOperations(dataset=rd, slice=[0, 0, 0, 0, 15], output_format=of, add_auxiliary_files=a)
            ret = ops.execute()
            with self.nc_scope(ret) as ds:
                self.assertEqual(len(ds.dimensions['nMesh2_face']), 1)
