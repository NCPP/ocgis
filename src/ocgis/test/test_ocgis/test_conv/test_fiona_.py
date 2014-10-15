from collections import OrderedDict
import os
import fiona
import ocgis
from ocgis.api.subset import SubsetOperation
from ocgis.conv.fiona_ import ShpConverter
from ocgis.test.base import TestBase
from ocgis.test.test_ocgis.test_api.test_parms.test_definition import TestGeom


class TestShpConverter(TestBase):

    def get_subset_operation(self):
        geom = TestGeom.get_geometry_dictionaries()
        rd = self.test_data_nc.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom=geom, select_nearest=True, snippet=True)
        subset = SubsetOperation(ops)
        return subset

    def test_attributes_copied(self):
        """Test attributes in geometry dictionaries are properly accounted for in the converter."""

        subset = self.get_subset_operation()
        conv = ShpConverter(subset, self.current_dir_output, prefix='shpconv')
        ret = conv.write()

        path_ugid = os.path.join(self.current_dir_output, conv.prefix+'_ugid.shp')

        with fiona.open(path_ugid) as source:
            self.assertEqual(source.schema['properties'], OrderedDict([(u'COUNTRY', 'str'), (u'UGID', 'int:10')]))

    def test_none_geom(self):
        """Test a NoneType geometry will pass through the Fiona converter."""

        rd = self.test_data_nc.get_rd('cancm4_tas')
        slc = [None, 0, None, [10, 20], [10, 20]]
        ops = ocgis.OcgOperations(dataset=rd, slice=slc)
        subset = SubsetOperation(ops)
        conv = ShpConverter(subset, self.current_dir_output, prefix='shpconv')
        ret = conv.write()
        contents = os.listdir(self.current_dir_output)
        self.assertEqual(len(contents), 5)
