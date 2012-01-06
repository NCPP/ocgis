import os
import datetime
import urllib
#import tempfile
#import django.test
from django.test import TestCase
from django.test.client import Client
#from django.test.testcases import TransactionTestCase
#from django.conf import settings
from models import *
from util.ncwrite import NcWrite, NcVariable, NcSpatial, NcTime
from util.helpers import get_temp_path
from shapely.geometry import Polygon


class UserGeometryMetadataTest(TestCase):
    
    fixtures = ['test_usgs-cida-maurer.json']

    def test_geom_gmap_static_url(self):
        
        url = UserGeometryMetadata.objects.get(pk=1).geom_gmap_static_url(
            color='0x00ff00', 
            weight=10
        )
        # try out the URL
        urllib.urlopen(url)
        
        self.assertEqual(
            url,
            'http://maps.googleapis.com/maps/api/staticmap'
            '?size=512x256&sensor=false'
            '&path=color:0x00ff00|weight:10'
            '|enc:_ilhE~f_cU?~oR_pR?~oR_pR'
            '&path=color:0x00ff00|weight:10'
            '|enc:_ilhE~|{|T?~oR_pR?~oR_pR'
        )

class UserGeometryDataTest(TestCase):
    
    fixtures = ['test_usgs-cida-maurer.json']
    
    def test_fixture_loading(self):
        '''Check that the test fixture loaded correctly'''
        self.assertEqual(NetcdfDataset.objects.count(), 1)
        
    def test_pathLocations(self):
        '''Test the output of a path Location string for a geometry'''
        from django.contrib.gis.geos import LineString
        
        self.assertEqual(
            UserGeometryData.objects.get(pk=1).pathLocations(),
            ['path=color:0x0000ff|weight:4|enc:_ilhE~f_cU?~oR_pR?~oR_pR']
        )


class ArchiveTest(TestCase):
    
    fixtures = ['test_usgs-cida-maurer.json']
    
    def test_metadata_list(self):
        test = Archive.objects.all()[0]  # get the first archive object
        self.assertEquals(test.metadata_list(), [])


class ScenarioTest(TestCase):
    
    fixtures = ['test_usgs-cida-maurer.json']
    
    def test_metadata_list(self):
        test = Scenario.objects.all()[0]  # get the first scenario object
#        # get the associate list of metadata
#        metadata_list = test.scenariometadataurl_set.model.objects.filter(scenario=test.pk)
        
        self.assertEquals(
            test.metadata_list(),
            ['External Metadata :: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Experiments']
        )


class ClimateModelTest(TestCase):
    
    fixtures = ['test_usgs-cida-maurer.json']
    
    def test_metadata_list(self):
        test = ClimateModel.objects.all()[0]  # get the first climate model object
        
        self.assertEquals(
            test.metadata_list(filter_field='model'),
            ['External Metadata :: http://www-pcmdi.llnl.gov/ipcc/model_documentation/BCC-CM1.htm',
             'External Metadata :: http://www.ipcc-data.org/ar4/model-BCC-CM1-change.html',
             'External Metadata :: http://bcc.cma.gov.cn/CSMD/en/']
        )


class VariableTest(TestCase):
    
    fixtures = ['test_usgs-cida-maurer.json']
    
    def test_metadata_list(self):
        test = Variable.objects.all()[0]  # get the first climate model object
        
        self.assertEquals(
            test.metadata_list(),
            ['External Metadata :: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Highest_priority_output']
        )

class NcwriteTest(TestCase):
    
    def test_write(self):
        
        ncvariable = NcVariable("Prcp","mm",constant=5)
        bounds = Polygon(((0,0),(10,0),(10,15),(0,15)))
        res = 5
        ncspatial = NcSpatial(bounds,res)
        rng = [datetime.datetime(2007,10,1),datetime.datetime(2007,10,3)]
        interval = datetime.timedelta(days=1)
        nctime = NcTime(rng,interval)
        ncw = NcWrite(ncvariable,ncspatial,nctime)
        path = get_temp_path(suffix='.nc')
        rootgrp = ncw.get_rootgrp(path)
        self.assertEquals(
            rootgrp.variables["Prcp"][:].shape,
            (3, 4, 3)
        )
        ncw = NcWrite(ncvariable,ncspatial,nctime,nlevels=4)
        path = get_temp_path(suffix='.nc')
        rootgrp = ncw.get_rootgrp(path)
        self.assertEquals(
            rootgrp.variables["Prcp"][:].shape,
            (3, 4, 4, 3)
        )
        ## check spatial dimensions
        self.assertEqual(ncw.ncspatial.dim_col.shape[0],3)
        self.assertEqual(ncw.ncspatial.dim_row.shape[0],4)
        ## write to a file
        ncw.write()


class ClimateDataTest(TestCase):
    
    def setUp(self):
        '''
        Set up the client for the tests
        '''
        # Every browser response test needs a client.
        self.client = Client()
    
    def test_index_page(self):
        """
        Tests that index page is rendered successfully;
        """
        response = self.client.get('/')
        self.failUnlessEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()