from django.test import TestCase
from util.ncconv import NetCdfAccessor
from util.ncwrite import NcWrite
from util.helpers import get_temp_path, parse_polygon_wkt
from climatedata.models import SpatialGridCell
import os
import climatedata
from django.contrib.gis.geos.collections import MultiPolygon
from django.contrib.gis.geos.polygon import Polygon
from django.test.client import Client
from util.toshp import OpenClimateShp


def get_fixtures():
    return [os.path.join(os.path.split(climatedata.__file__)[0],'fixtures','trivial_grid.json')]

def get_example_netcdf():
    var = 'Tavg'
    units_var = 'C'
    nw = NcWrite(var,units_var)
    path = get_temp_path(suffix='.nc')
    rootgrp = nw.write(path,close=False)
    return({
            'var':var,
            'units_var':units_var,
            'nw':nw,
            'path':path,
            'rootgrp':rootgrp,
            })


class NetCdfAccessTest(TestCase):
    """
    Tests requiring an NetCDF file to read should subclass this. Once a test
    OpenDap server is available, this object is obsolete.
    """
    
    fixtures = get_fixtures()
    
    def setUp(self):
        self.client = Client()
        
        attrs = get_example_netcdf()
        for key,value in attrs.iteritems():
            setattr(self,key,value)
#        self.var = 'Tavg'
#        self.units_var = 'C'
#        self.nw = NcWrite(self.var,self.units_var)
#        self.path = get_temp_path(suffix='.nc')
#        self.rootgrp = self.nw.write(self.path,close=False)
        
    def tearDown(self):
        self.rootgrp.close()
        
        
class TestUrls(NetCdfAccessTest):
    """Test URLs for correct response codes."""

    def test_archives(self):
        urls = [
                '/api/archives/',
                '/api/archives.html',
                '/api/archives.json',
                '/api/archives/cmip3/',
                '/api/archives/cmip3.html',
                '/api/archives/cmip3.json'
                ]
        for url in urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code,200)
        
        ## confirm the correct reponse code is raised
        response = self.client.get('/api/archives/bad_archive.json')
        self.assertEqual(response.status_code,404)
        
    def test_test_urls(self):
        ## test the spatial handler and zip response
        response = self.client.get('/api/test/shz/')
        self.assertEqual(response.status_code,200)
        
#        ## test spatial handler with a polygon intersection
#        Polygon(((11.5,3.5),(12.5,3.5),(12.5,2.5),(11.5,2.5),(11.5,3.5)))
        url = '/api/test/shz/?poly__11_5-3_5__12_5-3_5__12_5-2_5__11_5-2_5'
        response = self.client.get(url)
        self.assertEqual(response.status_code,200)


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
    
    def get_object(self):
        """Return an example OpenClimateShp object."""
        
        qs = SpatialGridCell.objects.all().order_by('row','col')
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        na = NetCdfAccessor(self.rootgrp,self.var)
        dl = na.get_dict(geom_list)
        path = get_temp_path('.shp')
        shp = OpenClimateShp(path,dl)
        return(shp)
    
    def test_write(self):
        """Write a shapefile."""
        
        shp = self.get_object()
        shp.write()
        
        
class TestHelpers(TestCase):
    
    def test_parse_polygon_wkt(self):
        """Test the parsing of the polygon query string."""
        
        actual = 'POLYGON ((30 10,10 20,20 40,40 40,30 10))'
        
        qs = ['poly__30-10__10-20__20-40__40-40',
             'POLYGON__30-10__10-20__20-40__40-40']
        
        for q in qs: 
            wkt = parse_polygon_wkt(q)
            self.assertEqual(wkt,actual)
            
        actual = 'POLYGON ((30 10,10 20,20 40,40 40,30 10))'