from django.test import TransactionTestCase, TestCase
from util.ncconv import NetCdfAccessor
from util.ncwrite import NcWrite
from util.helpers import get_temp_path
from climatedata.models import SpatialGridCell
import os
import climatedata
from django.contrib.gis.geos.collections import MultiPolygon
from django.contrib.gis.geos.polygon import Polygon
from django.test.client import Client
from util.toshp import OpenClimateShp


def get_fixtures():
    return [os.path.join(os.path.split(climatedata.__file__)[0],'fixtures','trivial_grid.json')]


class NetCdfAccessTest(TestCase):
    fixtures = get_fixtures()
    
    def setUp(self):
        self.client = Client()
        self.var = 'Tavg'
        self.units_var = 'C'
        self.nw = NcWrite(self.var,self.units_var)
        self.path = get_temp_path(suffix='.nc')
        self.rootgrp = self.nw.write(self.path,close=False)
        
    def tearDown(self):
        self.rootgrp.close()
        
        
class TestUrls(NetCdfAccessTest):
    
    def test_shapefile(self):
        response = self.client.get('/api/shz/')


class NetCdfAccessorTests(NetCdfAccessTest):
    
    def test_constructor(self):
        na = NetCdfAccessor(self.rootgrp,self.var)
        self.assertTrue(len(na._timevec) > 0)

    def test_get_dict(self):
        """Convert entire NetCDF to dict."""
        qs = SpatialGridCell.objects.all().order_by('row','col')
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        na = NetCdfAccessor(self.rootgrp,self.var)
        dl = na.get_dict(geom_list)
        self.assertEquals(len(dl),len(geom_list)*len(self.nw.dim_time))
        
    def test_get_dict_intersects(self):
        """Convert subset of NetCDF to dict."""
        igeom = Polygon(((11.5,3.5),(12.5,3.5),(12.5,2.5),(11.5,2.5),(11.5,3.5)))
        qs = SpatialGridCell.objects.filter(geom__intersects=igeom).order_by('row','col')
        y_indices = [obj.row for obj in qs]
        x_indices = [obj.col for obj in qs]
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        na = NetCdfAccessor(self.rootgrp,self.var)
        dl = na.get_dict(geom_list,x_indices=x_indices,y_indices=y_indices)
        self.assertEqual(len(dl),len(geom_list)*len(self.nw.dim_time))
        
        
class OpenClimateShpTests(NetCdfAccessTest):
    
    def test_constructor(self):
        qs = SpatialGridCell.objects.all().order_by('row','col')
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        na = NetCdfAccessor(self.rootgrp,self.var)
        dl = na.get_dict(geom_list)
        path = get_temp_path('.shp')
        shp = OpenClimateShp(path,dl)
        return(shp)
    
    def test_write(self):
        shp = self.test_constructor()
        shp.write()