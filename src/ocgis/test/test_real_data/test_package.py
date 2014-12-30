from ocgis.test.base import TestBase
import ocgis
from ocgis.exc import DefinitionValidationError
from ocgis import constants


class Test(TestBase):

    def test_nc(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_rhsmax')
        rd = [rd1, rd2]
        for output_format in [constants.OUTPUT_FORMAT_SHAPEFILE, constants.OUTPUT_FORMAT_CSV,
                              constants.OUTPUT_FORMAT_CSV_SHAPEFILE, constants.OUTPUT_FORMAT_NETCDF]:
            if output_format == constants.OUTPUT_FORMAT_NETCDF:
                with self.assertRaises(DefinitionValidationError):
                    ocgis.OcgOperations(dataset=rd, output_format=output_format, geom='state_boundaries',
                                        select_ugid=[25], snippet=True, prefix=output_format)
            else:
                ops = ocgis.OcgOperations(dataset=rd, output_format=output_format, geom='state_boundaries',
                                          select_ugid=[25], snippet=True, prefix=output_format)
                ops.execute()
