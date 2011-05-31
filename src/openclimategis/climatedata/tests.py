from django.test import TestCase
from django.test.client import Client
from models import *
from util.ncwrite import NcWrite

class NcwriteTest(TestCase):
    fixtures = ['trivial_grid.json']
    
    def test_write(self):
        nw = NcWrite()
        self.assertTrue(len(nw.centroids)>0)


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
    

