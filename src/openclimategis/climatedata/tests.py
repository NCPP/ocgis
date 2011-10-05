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

#def import_examples():
#    organization,created = Organization.objects.get_or_create(
#               name='National Center for Atmospheric Research',
#               code='NCAR',
#               country='USA',
#               url='http://ncar.ucar.edu/')
#    organization.save()
#    archive,created = Archive.objects.get_or_create(
#              name='Coupled Model Intercomparison Project - Phase 3',
#              code='CMIP3',
#              url='http://cmip-pcmdi.llnl.gov/cmip3_overview.html')
#    archive.save()
#    scenario,created = Scenario.objects.get_or_create(
#                       name='1 percent to 2x CO2',
#                       code='1pctto2x')
#    scenario.save()
#    
#    folder = os.path.join(settings.TEST_CLIMATE_DATA,'wcrp_cmip3')
#    uris = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.nc')]
#    kwds = dict(name='BCCR-BCM2.0',
#                code='BCCR-BCM2.0',
#                url='http://www-pcmdi.llnl.gov/ipcc/model_documentation/BCCR_BCM2.0.htm',
#                scenario_regex='pcmdi\.ipcc4\.bccr_bcm2_0\.(........)\..*nc')
#    load_climatemodel(archive,uris,**kwds)
#    
#    ## -------------------------------------------------------------------------
#    
#    archive,created = Archive.objects.get_or_create(
#              organization=organization,
#              name='Bias Corrected and Downscaled WCRP CMIP3 Climate',
#              code='Maurer07',
#              url='http://gdo-dcp.ucllnl.org/downscaled_cmip3_projections/')
#    archive.save()
#    
#    scenario,created = Scenario.objects.get_or_create(
#                       name='720 ppm stabilization experiment',
#                       code='sresa1b')
#    scenario.save()
#    scenario,created = Scenario.objects.get_or_create(
#                       name='SRES A2 experiment',
#                       code='sresa2')
#    scenario.save()
#    
#    folder = os.path.join(settings.TEST_CLIMATE_DATA,'maurer')
#    
#    uris = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.nc') and f.startswith('bccr')]
#    kwds = dict(name='BCCR-BCM2.0',
#                code='BCCR-BCM2.0',
#                url='http://www-pcmdi.llnl.gov/ipcc/model_documentation/BCCR_BCM2.0.htm',
#                scenario_regex='.*\.1\.(.*)\.monthly\.Prcp\.19..\.nc')
#    load_climatemodel(archive,uris,**kwds)
#    
#    uris = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.nc') and f.startswith('cccma')]
#    kwds = dict(name='CCCMA-CGCM3.1',
#                code='CCCMA-CGCM3.1',
#                url='',
#                scenario_regex='.*\.1\.(.*)\.monthly\.Prcp\.19..\.nc')
#    load_climatemodel(archive,uris,**kwds)
#
#
#class LoadData(TransactionTestCase):
#        
#    def test_load(self):
#        import_examples()
#        import pdb;pdb.set_trace()


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


#class InitialDataTest(TestCase):
#    def test_initial_data(self):
#        "Tests data installed from the initial_data fixture"
#        self.assertEquals(Scenario.objects.count(), 3)


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
