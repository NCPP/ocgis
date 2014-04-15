from ocgis.test.base import TestBase
import ocgis
from ocgis.api.subset import SubsetOperation
from ocgis.api.collection import SpatialCollection
import itertools
from ocgis.util.logging_ocgis import ProgressOcgOperations


class Test(TestBase):
    
    def get_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None,[0,100],None,[0,10],[0,10]]
        ops = ocgis.OcgOperations(dataset=rd,slice=slc)
        return(ops)
    
    def test_constructor(self):
        for rb,p in itertools.product([True,False],[None,ProgressOcgOperations()]):
            sub = SubsetOperation(self.get_operations(),request_base_size_only=rb,
                                  progress=p)
            for ii,coll in enumerate(sub):
                self.assertIsInstance(coll,SpatialCollection)
        self.assertEqual(ii,0)
