from django.test import TestCase
from django.test.client import Client
import subprocess
import itertools
from traceback import print_exc
import sys


def pause(f):
    print('test "{0}" paused...'.format(f.__name__))


class TestCdata(TestCase):
    fixtures = ['cdata.json']
    c = Client()
    
    def test_get_data(self):
        spaces = [
                 '-123.4|45.6|-122.2|48.7',
#                 '-128.66|44.27|-121.81|50.69'
#                 'mi_watersheds',
#                 'co_watersheds',
#                 'state_boundaries'
                 ]
        
        datasets = [
                    [1,'tasmax'],
#                    ['/home/local/WX/ben.koziol/links/ocgis/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc','cl'],
#                    [7,'rhsmax'],
                    ['4|9','tas|rhs']
                   ]
        
        outputs = [
                  'keyed',
#                  'meta',
#                  'nc',
                  ]
        
        times = [
                 'none',
#                 '2002-1-1|2002-12-31'
                 ]
        
        levels = ['none']
        
        operations = [
                      'intersects',
                      'clip'
                      ]
        
        aggregates = [
                      'true',
                      'false'
                      ]
        
        calcs = [
#                 'none',
                 'min~min_val|max~max_val',
#                 'heat_index~hi!tas~tas!rhs~rhs!units~K|min~min_val|max~max_val',
                 ]
        
        calc_raws = [
#                     'none',
                     'true',
                     'false'
                     ]
        
        calc_groupings = [
#                          'none',
                          'month',
#                          'day|month|year',
#                          'year',
#                          'year|month'
                          ]
        
        s_abstractions = [
                          'polygon',
#                          'point'
                          ]
        
#        calc = ''
#        calc = ('&calc=max~max|min~min&calc_raw=false&calc_grouping=month')
#        calc = ('&calc='
#                'max_cons~max_cons_gte!threshold~15!operation~gte|'
#                'max_cons~max_cons_lt!threshold~15!operation~lt'
#                '&calc_grouping=month|year&calc_raw=false')

        def _append_(url,key,value,prepend=True):
            kv = '{0}={1}'.format(key,value)
            if prepend:
                kv = '&'+kv
            return(url+kv)

        args = (spaces,datasets,outputs,times,levels,operations,aggregates,calcs,calc_raws,calc_groupings,s_abstractions)
        for space,dataset,output,time,level,operation,agg,calc,calc_raw,calc_grouping,s_abstraction in itertools.product(*args):
            dataset[0] = str(dataset[0])
            if '|' not in dataset[0] and 'heat_index' in calc:
                continue
            url = '/subset?'
            url = _append_(url,'space',space,prepend=False)
            if len(dataset[0]) > 10:
                url = _append_(url,'uri',dataset[0])
            else:
                url = _append_(url,'uid',dataset[0])
            url = _append_(url,'variable',dataset[1])
            url = _append_(url,'output',output)
            url = _append_(url,'time',time)
            url = _append_(url,'level',level)
            url = _append_(url,'operation',operation)
            url = _append_(url,'aggregate',agg)
            url = _append_(url,'calc',calc)
            url = _append_(url,'calc_raw',calc_raw)
            url = _append_(url,'calc_grouping',calc_grouping)
            url = _append_(url,'s_abstraction',s_abstraction)
            
#            try:
            resp = self.c.get(url)
#            except:
#                print_exc(file=sys.stdout)
#                import ipdb;ipdb.set_trace()
        
    def open_in_chrome(self,url):
        subprocess.call(["google-chrome",'http://127.0.0.1:8000'+url])
        
    def test_display_inspect(self):
        uid = '5'
        uri = '/usr/local/climate_data/CanCM4/tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
        
        url1 = '/inspect?uid='+uid
        url2 = '/inspect?uri='+uri
        
        for url in [url1,url2]:
            resp = self.c.get(url)
            self.assertTrue(len(resp.content) > 100)
        
    def test_get_shp(self):
#        url = '/shp/co_watersheds'
        url = '/shp/state_boundaries'
#        self.open_in_chrome(url)
        resp = self.c.get(url)
        
    def test_get_snippet(self):
        uids = ['3']
#        uids = ['/home/local/WX/ben.koziol/links/ocgis/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc']
        variables = [
                     'tas',
#                     'cl'
                     ]
        prefixes = [
#                    'none',
                    'some_prefix'
                    ]
        spaces = [
                 'none',
                 '-123.4|45.6|-122.2|48.7',
                 'mi_watersheds',
                 'co_watersheds',
                 'state_boundaries'
                 ]
        
        for uid,variable,prefix,space in itertools.product(uids,variables,prefixes,spaces):
            if len(uid) > 10:
                url = '/snippet?uri={uid}&variable={variable}&prefix={prefix}&space={space}'
            else:
                url = '/snippet?uid={uid}&variable={variable}&prefix={prefix}&space={space}'
            url = url.format(uid=uid,variable=variable,prefix=prefix,space=space)
            resp = self.c.get(url)
#        self.open_in_chrome(url)

    def test_point_abstraction(self):
        url = '/snippet?uid=3&variable=tas&s_abstraction=point'
        resp = self.c.get(url)
        import ipdb;ipdb.set_trace()
    
    @pause
    def test_multi_url(self):
        url = ('/uid/1|2/variable/tasmax/level/none/time/none/space/state_boundaries'
               '/operation/clip/aggregate/true/output/keyed')
        resp = self.c.get(url)
    
    @pause
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
    
    @pause
    def test_multivariate_request(self):
        url = ('/uid/4|9/variable/tas|rhs/level/none/time/none/space/state_boundaries'
               '/operation/intersects/aggregate/false/output/keyed'
               '?calc=heat_index~hi!tas~tas!rhs~rhs!units~K&calc_grouping=day|month|year')
        resp = self.c.get(url)
    
    @pause
    def test_presentation_urls(self):
        
        fus = []
        def _fu(url):
            fus.append('http://127.0.0.1:8000'+url)
            return(url)
        
#        ## inspect url
#        url = _fu('/inspect/uid/none?uri=/usr/local/climate_data/CanCM4/tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc')
#        resp = self.c.get(url)
#        self.assertEqual(len(resp.content),4842)
#        
#        ## snippet url
#        url = _fu('/snippet/uid/5/variable/tasmin?prefix=snippet_tasmin')
#        resp = self.c.get(url)
#        self.assertEqual(resp.tell(),285437)
#        self.assertTrue(resp._headers['content-disposition'][1].startswith('attachment; filename=snippet_tasmin_'))
#        
#        ## download shapefiles
#        keys = ['co_watersheds','state_boundaries']
#        for key in keys:
#            url = _fu('/shp/{0}'.format(key))
#            resp = self.c.get(url)
#            self.assertEqual(resp._headers['content-disposition'][1],'attachment; filename={0}.zip'.format(key))
#            
#        ## basic overlays
#        keys = ['co_watersheds','state_boundaries']
#        for key in keys:
#            url = _fu('/snippet/uid/5/variable/tasmin?prefix=snippet_{0}&space={0}'.format(key))
#            resp = self.c.get(url)
#            self.assertTrue(int(resp._headers['content-length'][1]) > 0)
#
#        ## co watersheds data download
#        url = ('/uid/5/variable/tasmin/level/none/time/none/'
#               'space/co_watersheds/operation/clip/aggregate/true/output/keyed'
#               '?prefix=tasmin_CanCM4')
#        resp = self.c.get(_fu(url))
#        
#        ## co watersheds calculation
#        outputs = ['meta','keyed']
#        for output in outputs:
#            url = ('/uid/6/variable/tasmin/level/none/time/none/'
#                   'space/co_watersheds/operation/clip/aggregate/true/output/{0}'
#                   '?prefix=calc_tasmin_CanCM4'
#                   '&calc=max_cons~max_cons_lte_0c_tasmin!threshold~273.175!operation~lte'
#                   '|mean~mean_tasmin'
#                   '|std~std_tasmin'
#                   '|min~min_tasmin'
#                   '|max~max_tasmin'
#                   '&calc_grouping=month|year&calc_raw=false'.format(output))
#            resp = self.c.get(_fu(url))
#        
#        ## heat index by state
#        url = ('/uid/4|9/variable/tas|rhs/level/none/time/none/space/state_boundaries'
#               '/operation/intersects/aggregate/false/output/meta'
#               '?calc=heat_index~hi!tas~tas!rhs~rhs!units~K&calc_grouping=day|month|year')
#        resp = self.c.get(_fu(url))
        
        ## selection by bounding box
        url = ('/uid/5/variable/tasmin/level/none/time/none/'
               'space/-109.06128|36.94111|-101.9751|41.07107/operation/clip/aggregate/false/output/keyed'
               '?prefix=tasmin_CanCM4_co_bb')
        resp = self.c.get(_fu(url))
        
        for fu in fus:
            print fu
