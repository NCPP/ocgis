import os
from ocgis import OcgOperations
from ocgis.exc import ExtentError
from ocgis.test.base import TestBase
from ocgis.util.itester import itr_products_keywords


class TestOcgInterpreter(TestBase):

    def test_execute_directory(self):
        """Test that the output directory is removed appropriately following an operations failure."""

        kwds = dict(add_auxiliary_files=[True, False])
        rd = self.test_data_nc.get_rd('cancm4_tas')

        ## this geometry is outside the domain and will result in an exception
        geom = [1000, 1000, 1100, 1100]

        for k in itr_products_keywords(kwds, as_namedtuple=True):
            ops = OcgOperations(dataset=rd, output_format='csv', add_auxiliary_files=k.add_auxiliary_files, geom=geom)
            try:
                ret = ops.execute()
            except ExtentError:
                contents = os.listdir(self.current_dir_output)
                self.assertEqual(len(contents), 0)
