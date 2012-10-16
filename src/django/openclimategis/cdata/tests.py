from django.test import TestCase
from django.test.client import Client
import os.path
import subprocess


CLIMATE_DATA = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data'
ALBISCCP = os.path.join(CLIMATE_DATA,'cmip5/albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc')

class CdataTest(TestCase):
    c = Client()
    
    def test_get_data(self):
#        space = '-123.4|45.6|-122.2|48.7'
        space = 'mi_watersheds'
        
#        uri = 'http://an.opendap.dataset'
        uri = ALBISCCP
        
        variable = 'albisccp'
        
#        output = 'keyed'
        output = 'meta'
#        output = 'csv'
        
        time = '0020-1-1|0020-12-31'
#        time = 'none'
        
#        calc = ''
        calc = ('&calc=max~max|min~min&calc_raw=false&calc_grouping=month')
#        calc = ('&calc='
#                'max_cons~max_cons_gte!threshold~15!operation~gte|'
#                'max_cons~max_cons_lt!threshold~15!operation~lt'
#                '&calc_grouping=month&calc_raw=false')
        
        url = ('/uid/none/variable/{variable}/level/none/time/{time}/'
               'space/{space}/operation/clip/aggregate/true/output/{output}'
               '?uri={uri}{calc}')
        url = url.format(space=space,
                         uri=uri,
                         variable=variable,
                         calc=calc,
                         output=output,
                         time=time)
        
#        self.open_in_chrome(url)
        resp = self.c.get(url)
        print resp.content
        
    def open_in_chrome(self,url):
        subprocess.call(["google-chrome",'http://127.0.0.1:8000'+url])
        
    def test_display_inspect(self):
        uri = ALBISCCP
        
        url = '/inspect/uid/none?uri={uri}'.format(uri=uri)
        
        resp = self.c.get(url)
        
    def test_get_shp(self):
        url = '/shp/co_watersheds'
        self.open_in_chrome(url)
#        resp = self.c.get(url)
