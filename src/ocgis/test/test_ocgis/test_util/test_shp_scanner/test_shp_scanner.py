from unittest.case import SkipTest

# this module is deprecated
raise SkipTest('module deprecated')

from ocgis.test.base import TestBase
from ocgis.util.shp_scanner.shp_scanner import get_does_intersect,\
    get_select_ugids, build_database, write_json
from ocgis.util.shp_cabinet import ShpCabinetIterator, ShpCabinet
from ocgis.api.operations import OcgOperations
from ocgis.api.request.base import RequestDataset
import tempfile
import json


class Test(TestBase):
    
    @property
    def nevada(self):
        return(self.get_geometry(23))
    
    @property
    def new_york(self):
        return(self.get_geometry(17))
    
    def get_geometry(self,select_ugid):
        geoms = list(ShpCabinetIterator(key='state_boundaries',select_ugid=[select_ugid]))
        return(geoms[0]['geom'])
    
    def get_subset_rd(self):
        rd = self.test_data_nc.get_rd('cancm4_tas')
        ret = OcgOperations(dataset=rd,geom=self.nevada,snippet=True,output_format='nc').execute()
        rd_sub = RequestDataset(uri=ret,variable='tas')
        return(rd_sub)
    
    def test_get_does_intersect_true(self):
        rd = self.test_data_nc.get_rd('cancm4_tas')
        for geom in [self.nevada,self.new_york]:
            self.assertTrue(get_does_intersect(rd,geom))
            
    def test_get_does_intersect_false(self):
        rd_sub = self.get_subset_rd()
        self.assertFalse(get_does_intersect(rd_sub,self.new_york))

    def test_get_select_ugids(self):
        path = ShpCabinet().get_shp_path('state_boundaries')
        rd = self.get_subset_rd()
        select_ugids = get_select_ugids(rd,path)
        for select_ugid in select_ugids:
            self.assertTrue(get_does_intersect(rd,self.get_geometry(select_ugid)))

    def test_build_database(self):
        from ocgis.util.shp_scanner.labels import StateBoundaries
        keys = {'state_boundaries':['US State Boundaries',StateBoundaries]}
        build_database(keys=keys,filter_request_dataset=self.get_subset_rd())
        handle,path = tempfile.mkstemp(suffix='.json',dir=self.current_dir_output)
        write_json(path)
        with open(path,'r') as fp:
            data = json.load(fp)
        self.assertEqual(data,{u'US State Boundaries': {u'geometries': {u'null': {u'Utah (UT)': 24, u'Nevada (NV)': 23, u'Idaho (ID)': 9, u'California (CA)': 25, u'Arizona (AZ)': 37, u'Oregon (OR)': 12}}, u'key': u'state_boundaries'}})
