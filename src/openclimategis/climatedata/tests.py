from django.test import TestCase
from django.test.client import Client

class InitialDataTest(TestCase):
    def test_initial_data(self):
        "Tests data installed from the intial_data fixture"
        from openclimategis.climatedata.models import Calendar
        
        self.assertEquals(Calendar.objects.count(), 10)


class TrivialGridTest(TestCase):
    fixtures = [
        'trivial_grid.json',
        'trivial_temporal_grid.json',
    ]
    
    def testTrivialSpatialGridFixture(self):
        "Tests that the spatial grid fixture data was loaded"
        from openclimategis.climatedata.models import SpatialGrid
        from openclimategis.climatedata.models import SpatialGridCell
        
        self.assertEquals(SpatialGrid.objects.count(), 1)
        self.assertEquals(SpatialGridCell.objects.count(), 12)
    
    def testTrivialTemporalGridFixture(self):
        "Tests that the temporal grid fixture data was loaded"
        from openclimategis.climatedata.models import TemporalUnit
        from openclimategis.climatedata.models import TemporalGrid
        from openclimategis.climatedata.models import TemporalGridCell
        
        self.assertEquals(TemporalUnit.objects.count(), 1)
        self.assertEquals(TemporalGrid.objects.count(), 1)
        self.assertEquals(TemporalGridCell.objects.count(), 3)


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

