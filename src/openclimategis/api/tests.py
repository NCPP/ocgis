from django.test import TestCase
import os
from netCDF4 import Dataset
from api.util.ncconv import GeoQuerySetFactory
from climatedata.models import Grid, GridCell


class LocalNetCdfTests(TestCase):
    bin = '/home/bkoziol/git/OpenClimateGIS/bin'
    nc = 'cnrm_cm3.1.sresa2.monthly.Tavg.2012.nc'
    var = 'Tavg'
    ncpath = os.path.join(bin,nc)


class GeoQuerySetFactoryTests(LocalNetCdfTests):
    
    def test_constructor(self):
        rootgrp = Dataset(self.ncpath,'r')
        gf = GeoQuerySetFactory(rootgrp,self.var)
        
        ## check time layers are created
        self.assertTrue(len(gf._timevec) > 0)
        
        rootgrp.close()
        
    def test_get_queryset(self):
        rootgrp = Dataset(self.ncpath,'r')
        gf = GeoQuerySetFactory(rootgrp,self.var)
        
        qs = GridCell.objects.all()
        for obj in qs:
            import ipdb;ipdb.set_trace()
        
        rootgrp.close()
        

__test__ = {"doctest": """
Another way to test that 1 + 1 is equal to 2.

>>> 1 + 1 == 2
True
"""}

