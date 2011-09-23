from django.test import TestCase
from django.test.client import Client
from models import *
from util.ncwrite import NcWrite
import tempfile
from util.helpers import get_temp_path
from django.test.testcases import TransactionTestCase
from django.conf import settings
import os
import django.test


def import_examples():
    organization,created = Organization.objects.get_or_create(
               name='National Center for Atmospheric Research',
               code='NCAR',
               country='USA',
               url='http://ncar.ucar.edu/')
    organization.save()
    archive,created = Archive.objects.get_or_create(
              organization=organization,
              name='Coupled Model Intercomparison Project - Phase 3',
              code='CMIP3',
              url='http://cmip-pcmdi.llnl.gov/cmip3_overview.html')
    archive.save()
    scenario,created = Scenario.objects.get_or_create(
                       name='1 percent to 2x CO2',
                       code='1pctto2x')
    scenario.save()
    
    folder = os.path.join(settings.TEST_CLIMATE_DATA,'wcrp_cmip3')
    uris = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.nc')]
    kwds = dict(name='BCCR-BCM2.0',
                code='BCCR-BCM2.0',
                url='http://www-pcmdi.llnl.gov/ipcc/model_documentation/BCCR_BCM2.0.htm',
                scenario_regex='pcmdi\.ipcc4\.bccr_bcm2_0\.(........)\..*nc')
    load_climatemodel(archive,uris,**kwds)
    
    ## -------------------------------------------------------------------------
    
    archive,created = Archive.objects.get_or_create(
              organization=organization,
              name='Bias Corrected and Downscaled WCRP CMIP3 Climate',
              code='Maurer07',
              url='http://gdo-dcp.ucllnl.org/downscaled_cmip3_projections/')
    archive.save()
    
    scenario,created = Scenario.objects.get_or_create(
                       name='720 ppm stabilization experiment',
                       code='sresa1b')
    scenario.save()
    scenario,created = Scenario.objects.get_or_create(
                       name='SRES A2 experiment',
                       code='sresa2')
    scenario.save()
    
    folder = os.path.join(settings.TEST_CLIMATE_DATA,'maurer')
    
    uris = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.nc') and f.startswith('bccr')]
    kwds = dict(name='BCCR-BCM2.0',
                code='BCCR-BCM2.0',
                url='http://www-pcmdi.llnl.gov/ipcc/model_documentation/BCCR_BCM2.0.htm',
                scenario_regex='.*\.1\.(.*)\.monthly\.Prcp\.19..\.nc')
    load_climatemodel(archive,uris,**kwds)
    
    uris = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.nc') and f.startswith('cccma')]
    kwds = dict(name='CCCMA-CGCM3.1',
                code='CCCMA-CGCM3.1',
                url='',
                scenario_regex='.*\.1\.(.*)\.monthly\.Prcp\.19..\.nc')
    load_climatemodel(archive,uris,**kwds)


class LoadData(TransactionTestCase):
        
    def test_load(self):
        import_examples()
        import pdb;pdb.set_trace()


class NcwriteTest(TestCase):
    fixtures = ['trivial_example.json']
    
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


class InitialDataTest(TestCase):
    def test_initial_data(self):
        "Tests data installed from the intial_data fixture"

        self.assertEquals(Calendar.objects.count(), 10)


class TrivialGridTest(TestCase):
    fixtures = [
        'trivial_example.json',
    ]
    
    def testTrivialGridFixture(self):
        "Tests that the grid fixture data was loaded"
        
        self.assertEquals(SpatialGrid.objects.count(), 1)
        self.assertEquals(SpatialGridCell.objects.count(), 12)
    
    def testTrivialTemporalGridFixture(self):
        "Tests that the temporal grid fixture data was loaded"
        
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

#class TestUrls(django.test.TestCase):
#    """Test URLs for correct response codes."""
#    
#    fixtures = ['luca_fixtures.json']
#    
#    def setUp(self):
#        self.client = Client()
#    
#    def test_urls(self):
#        ext = 'geojson'
#        drange = '2010-3-1+2010-4-30'
#        polygon = '-96+38,-95+38,-95+39,-96+39'
#        sop = 'clip'
#        agg = 'false'
#        cm = 'bccr_bcm2.0'
#        scenario = 'sresa1b'
#        archive = 'maurer'
#        var = 'prcp'
#        
#        base_url = ('/api/test/archive/{archive}/model/{cm}/scenario/{scenario}/'
#                    'temporal/{drange}/spatial/{sop}+polygon'
#                    '(({polygon}))/aggregate/{agg}/'
#                    'variable/{variable}.{ext}')
#        
#        url = base_url.format(ext=ext,
#                              drange=drange,
#                              polygon=polygon,
#                              sop=sop,
#                              agg=agg,
#                              cm=cm,
#                              scenario=scenario,
#                              archive=archive,
#                              variable=var)
#        
#        response = self.client.get(url)