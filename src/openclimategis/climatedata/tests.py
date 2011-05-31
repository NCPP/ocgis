from django.test import TestCase
from django.test.client import Client
from models import *
from util.ncwrite import NcWrite
import tempfile
from util.helpers import get_temp_path

class NcwriteTest(TestCase):
    fixtures = ['trivial_grid.json']
    
    def test_write(self):
        nw = NcWrite('Tavg','C')
        ## confirm the centroid coordinates are returned properly
        for c in nw.centroids:
            self.assertEquals(len(c.centroid.coords),2)
        ## check dimensions
        self.assertTrue(nw.dim_x.shape[0] > 0)
        self.assertTrue(nw.dim_y.shape[0] > 0)
        ## write to a file
        path = get_temp_path(suffix='.nc')
        print('test_write NC = {0}'.format(path))


class TrivialGridTest(TestCase):
    fixtures = ['trivial_grid.json']
    
    def testTrivialGridFixture(self):
        "Tests that the grid fixture data was loaded"
        
        self.assertEquals(SpatialGrid.objects.count(), 1)
        self.assertEquals(SpatialGridCell.objects.count(), 12)


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
    

