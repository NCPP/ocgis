from django.test import TestCase
from django.test.client import Client
import os.path
import subprocess


CLIMATE_DATA = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data'
ALBISCCP = os.path.join(CLIMATE_DATA,'cmip5/albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc')

class CdataTest(TestCase):
    fixtures = ['cdata.json']
    c = Client()
    
    def test_get_data(self):
        space = '-123.4|45.6|-122.2|48.7'
#        space = 'mi_watersheds'
#        space = 'co_watersheds'
        
#        uri = 'http://an.opendap.dataset'
        uri = ALBISCCP
        
        variable = 'albisccp'
        
        output = 'keyed'
#        output = 'meta'
#        output = 'csv'
        
#        time = '0020-1-1|0020-12-31'
#        time = '0020-1-1|0020-1-31'
        time = '0020-1-1|0049-12-13'
#        time = 'none'
        
#        calc = ''
#        calc = ('&calc=max~max|min~min&calc_raw=false&calc_grouping=month')
        calc = ('&calc='
                'max_cons~max_cons_gte!threshold~15!operation~gte|'
                'max_cons~max_cons_lt!threshold~15!operation~lt'
                '&calc_grouping=month|year&calc_raw=false')
        
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
#        print resp.content
        
    def open_in_chrome(self,url):
        subprocess.call(["google-chrome",'http://127.0.0.1:8000'+url])
        
    def test_display_inspect(self):
        uri = ALBISCCP
        
        url = '/inspect/uid/none?uri={uri}'.format(uri=uri)
        
        resp = self.c.get(url)
        
    def test_get_shp(self):
#        url = '/shp/co_watersheds'
        url = '/shp/state_boundaries'
#        self.open_in_chrome(url)
        resp = self.c.get(url)
        
    def test_get_snippet(self):
#        uri = ALBISCCP
#        uid = 'none'
        uri = ''
        uid = 3
        variable = 'tas'
        query = '?prefix=my_prefix'
        url = '/snippet/uid/{uid}/variable/{variable}{uri}{query}'\
              .format(uri=uri,uid=uid,variable=variable,query=query)
        resp = self.c.get(url)
#        self.open_in_chrome(url)
        
    def test_multi_url(self):
        url = ('/uid/1|2/variable/tasmax/level/none/time/none/space/state_boundaries'
               '/operation/clip/aggregate/true/output/keyed')
        resp = self.c.get(url)
        
    def test_multi_calc_url(self):
        url = ('/uid/1|2/variable/tasmax/level/none/time/none/space/co_watersheds'
               '/operation/clip/aggregate/true/output/meta'
               '?calc=max_cons~max_cons_gte!threshold~305.372!operation~gte'
               '|max_cons~max_cons_lt!threshold~273.15!operation~lte'
               '|mean~mean_temp'
               '|std~std_temp'
               '|min~min_temp'
               '|max~max_temp'
               '&calc_grouping=month|year&calc_raw=false')
        resp = self.c.get(url)
    
    def test_multivariate_request(self):
        url = ('/uid/4|9/variable/tas|rhs/level/none/time/none/space/state_boundaries'
               '/operation/intersects/aggregate/false/output/keyed'
               '?calc=heat_index~hi!tas~tas!rhs~rhs!units~K&calc_grouping=day|month|year')
        resp = self.c.get(url)
