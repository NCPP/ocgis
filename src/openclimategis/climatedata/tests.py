from django.test import TestCase
from django.test.client import Client

class SimpleTest(TestCase):
    def test_basic_addition(self):
        """
        Tests that 1 + 1 always equals 2.
        """
        self.failUnlessEqual(1 + 1, 2)


class TrivialGridTest(TestCase):
    fixtures = ['trivial_grid.json']
    
    def testTrivialGridFixture(self):
        "Tests that the grid fixture data was loaded"
        from openclimategis.climatedata.models import Grid, GridCell
        
        self.assertEquals(Grid.objects.count(), 1)
        self.assertEquals(GridCell.objects.count(), 12)


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
    

