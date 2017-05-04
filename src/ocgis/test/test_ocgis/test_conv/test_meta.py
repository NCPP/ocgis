import os
from copy import deepcopy

from ocgis import OcgOperations
from ocgis import constants
from ocgis.conv.meta import MetaOCGISConverter, MetaJSONConverter, AbstractMetaConverter
from ocgis.exc import DefinitionValidationError
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class TestMetaJSONConverter(TestBase):
    def get(self):
        ops = self.get_operations()
        return MetaJSONConverter(ops)

    def get_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd, output_format=constants.OUTPUT_FORMAT_METADATA_JSON)
        return ops

    @attr('data')
    def test_init(self):
        self.assertEqual(MetaJSONConverter.__bases__, (AbstractMetaConverter,))
        self.get()

    @attr('data')
    def test_operations(self):
        ops = self.get_operations()
        self.assertIsInstance(ops.execute(), basestring)

    @attr('data')
    def test_validate_ops(self):
        rd = self.test_data.get_rd('cancm4_tas')

        # Test only one request dataset allowed for metadata JSON output.
        rd2 = deepcopy(rd)
        rd2.alias = 'foo'
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=[rd, rd2], output_format=constants.OUTPUT_FORMAT_METADATA_JSON)

        # Test fields are not convertible to metadata JSON.
        field = rd.get()
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=field, output_format=constants.OUTPUT_FORMAT_METADATA_JSON)

    @attr('data')
    def test_write(self):
        mj = self.get()
        ret = mj.write()
        self.assertIsInstance(ret, basestring)


class TestMetaOCGISConverter(TestBase):
    @attr('data')
    def test_init(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        MetaOCGISConverter(ops)

    @attr('data')
    def test_write(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        conv = MetaOCGISConverter(ops)
        self.assertTrue(len(conv.write()) > 4000)
        self.assertEqual(len(os.listdir(self.current_dir_output)), 0)
