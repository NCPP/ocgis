from copy import deepcopy

from ocgis import OcgOperations
from ocgis.test.base import TestBase, attr


class Test(TestBase):
    @attr('slow')
    def test_many_request_datasets(self):
        """Test numerous request datasets."""

        rd_base = self.test_data.get_rd('cancm4_tas')
        geom = [-74.0, 40.0, -72.0, 42.0]
        rds = [deepcopy(rd_base) for ii in range(500)]
        for rd in rds:
            ret = OcgOperations(dataset=rd, geom=geom, snippet=True).execute()
            actual = ret.get_element(variable_name='tas').shape
            self.assertEqual(actual, (1, 2, 1))
