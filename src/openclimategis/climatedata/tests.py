import os
import datetime
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
