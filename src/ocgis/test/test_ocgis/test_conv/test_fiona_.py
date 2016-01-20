import datetime
import os
from collections import OrderedDict

import fiona
import numpy as np

import ocgis
from ocgis.api.subset import SubsetOperation
from ocgis.conv.fiona_ import ShpConverter, AbstractFionaConverter
from ocgis.interface.base.crs import WGS84
from ocgis.test.base import TestBase
from ocgis.test.base import attr
from ocgis.test.test_ocgis.test_api.test_parms.test_definition import TestGeom


class TestShpConverter(TestBase):
    def get_subset_operation(self):
        geom = TestGeom.get_geometry_dictionaries()
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom=geom, select_nearest=True, snippet=True)
        subset = SubsetOperation(ops)
        return subset

    def test_init(self):
        field = self.get_field()
        coll = field.as_spatial_collection()
        conv = ShpConverter([coll], outdir=self.current_dir_output, prefix='foo')
        self.assertIsInstance(conv, AbstractFionaConverter)
        self.assertFalse(conv.melted)

        conv = ShpConverter([coll], outdir=self.current_dir_output, prefix='foo', melted=True)
        self.assertTrue(conv.melted)

    @attr('data')
    def test_attributes_copied(self):
        """Test attributes in geometry dictionaries are properly accounted for in the converter."""

        subset = self.get_subset_operation()
        conv = ShpConverter(subset, outdir=self.current_dir_output, prefix='shpconv')
        ret = conv.write()

        path_ugid = os.path.join(self.current_dir_output, conv.prefix + '_ugid.shp')

        with fiona.open(path_ugid) as source:
            self.assertEqual(source.schema['properties'], OrderedDict([(u'COUNTRY', 'str:80'), (u'UGID', 'int:10')]))

    def test_build(self):
        field = self.get_field()
        coll = field.as_spatial_collection()
        conv = ShpConverter([coll], outdir=self.current_dir_output, prefix='foo')
        # no coordinate system...
        with self.assertRaises(ValueError):
            conv._build_(coll)

        field = self.get_field(crs=WGS84())
        coll = field.as_spatial_collection()
        conv = ShpConverter([coll], outdir=self.current_dir_output, prefix='foo')
        self.assertTrue(conv._use_upper_keys)
        ret = conv._build_(coll)
        schema_keys = ret['fobject'].meta['schema']['properties'].keys()
        for key in schema_keys:
            self.assertFalse(key.islower())
        self.assertNotIn('VALUE', ret['schema']['properties'])

        field = self.get_field(crs=WGS84())
        coll = field.as_spatial_collection()
        conv = ShpConverter([coll], outdir=self.current_dir_output, prefix='foo2', melted=True)
        ret = conv._build_(coll)
        self.assertIn('VALUE', ret['schema']['properties'])

    def test_get_field_type(self):
        target = ShpConverter.get_field_type(np.int32)
        self.assertEqual(target, 'int')
        key = 'foo'
        fiona_conversion = {}
        ShpConverter.get_field_type(np.int32, key=key, fiona_conversion=fiona_conversion)
        self.assertEqual(fiona_conversion[key], int)

        target = ShpConverter.get_field_type(str)
        self.assertEqual(target, 'str')

        target = ShpConverter.get_field_type(datetime.datetime)
        self.assertEqual(target, 'str')

        the_type = np.dtype('S20')
        target = ShpConverter.get_field_type(the_type)
        self.assertEqual(target, 'str:20')
        key = 'hey'
        fiona_conversion = {}
        ShpConverter.get_field_type(the_type, key=key, fiona_conversion=fiona_conversion)
        self.assertEqual(fiona_conversion[key], unicode)

    @attr('data')
    def test_none_geom(self):
        """Test a NoneType geometry will pass through the Fiona converter."""

        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None, 0, None, [10, 20], [10, 20]]
        ops = ocgis.OcgOperations(dataset=rd, slice=slc)
        subset = SubsetOperation(ops)
        conv = ShpConverter(subset, outdir=self.current_dir_output, prefix='shpconv')
        conv.write()
        contents = os.listdir(self.current_dir_output)
        self.assertEqual(len(contents), 5)

    def test_write_coll(self):

        def _test_key_case_(path, upper=True):
            with fiona.open(path, 'r') as source:
                for row in source:
                    keys = row['properties'].keys()
                    for key in keys:
                        if upper:
                            self.assertTrue(key.isupper(), key)
                        else:
                            self.assertFalse(key.isupper(), key)

        field = self.get_field(crs=WGS84())
        coll = field.as_spatial_collection()
        conv = ShpConverter([coll], outdir=self.current_dir_output, prefix='foo')
        f = conv._build_(coll)
        conv._write_coll_(f, coll)
        conv._finalize_(f)
        _test_key_case_(conv.path, upper=True)

        field = self.get_field(crs=WGS84())
        coll = field.as_spatial_collection()
        conv = ShpConverter([coll], outdir=self.current_dir_output, prefix='foo2', melted=True)
        f = conv._build_(coll)
        conv._write_coll_(f, coll)
        conv._finalize_(f)
        _test_key_case_(conv.path, upper=True)