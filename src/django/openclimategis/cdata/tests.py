from django.test import TestCase
from django.test.client import Client
import subprocess
import itertools
from ocgis.api.operations import OcgOperations
from ocgis import env
import time
from ocgis.api.interpreter import OcgInterpreter
from cdata.models import Address
import re
import os.path


class Test(TestCase):
    fixtures = ['cdata.json']
    c = Client()
    
    def iter_datasets(self):
        for address in Address.objects.all():
            uri = address.uri
            variable = os.path.split(uri)[1].split('_')[0]
            yield({'uri':uri,'variable':variable})
            
    def test_subset(self):
        for dataset in self.iter_datasets():
            ops = OcgOperations(dataset,snippet=True,output_format='keyed')
            url = ops.as_url()
            ret = self.c.get(url)
            self.assertTrue(len(ret.content) > 100)

#def pause(f):
#    print('test "{0}" paused...'.format(f.__name__))
#
#
#class TestCdata(TestCase):
#    fixtures = ['cdata.json']
#    c = Client()
#    
#    def test_get_data(self):
#        snippets = [
##                   'none',
##                   'true',
#                   'false'
#                   ]
#        spaces = [
##                 '-123.4|45.6|-122.2|48.7',
##                 '-128.66|44.27|-121.81|50.69'
#                 'mi_watersheds',
##                 'co_watersheds',
##                 'state_boundaries'
#                 ]
#        
#        datasets = [
#                    [1,'tasmax'],
##                    ['/home/local/WX/ben.koziol/links/ocgis/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc','cl'],
##                    [7,'rhsmax'],
##                    ['4|9','tas|rhs']
#                   ]
#        
#        outputs = [
#                   'shp'
##                  'keyed',
##                  'meta',
##                  'nc',
#                  ]
#        
#        times = [
#                 'none',
##                 '2002-1-1|2002-12-31'
#                 ]
#        
#        levels = ['none']
#        
#        operations = [
##                      'intersects',
#                      'clip'
#                      ]
#        
#        aggregates = [
#                      'true',
##                      'false'
#                      ]
#        
#        calcs = [
##                 'none',
#                 'min~min_val|max~max_val',
##                 'heat_index~hi!tas~tas!rhs~rhs!units~K|min~min_val|max~max_val',
#                 ]
#        
#        calc_raws = [
##                     'none',
##                     'true',
#                     'false'
#                     ]
#        
#        calc_groupings = [
##                          'none',
#                          'month',
##                          'day|month|year',
##                          'year',
##                          'year|month'
#                          ]
#        
#        s_abstractions = [
#                          'polygon',
##                          'point'
#                          ]
#        
#        agg_selections = [
##                          'none',
##                          'false',
#                          'true'
#                          ]
#        
##        calc = ''
##        calc = ('&calc=max~max|min~min&calc_raw=false&calc_grouping=month')
##        calc = ('&calc='
##                'max_cons~max_cons_gte!threshold~15!operation~gte|'
##                'max_cons~max_cons_lt!threshold~15!operation~lt'
##                '&calc_grouping=month|year&calc_raw=false')
#
#        def _append_(url,key,value,prepend=True):
#            kv = '{0}={1}'.format(key,value)
#            if prepend:
#                kv = '&'+kv
#            return(url+kv)
#
#        args = (spaces,datasets,outputs,times,levels,operations,aggregates,calcs,calc_raws,calc_groupings,s_abstractions,snippets,agg_selections)
#        for space,dataset,output,time,level,operation,agg,calc,calc_raw,calc_grouping,s_abstraction,snippet,agg_selection in itertools.product(*args):
#            dataset[0] = str(dataset[0])
#            if '|' not in dataset[0] and 'heat_index' in calc:
#                continue
#            url = '/subset?'
#            url = _append_(url,'geom',space,prepend=False)
#            if len(dataset[0]) > 10:
#                url = _append_(url,'uri',dataset[0])
#            else:
#                url = _append_(url,'uid',dataset[0])
#            url = _append_(url,'variable',dataset[1])
#            url = _append_(url,'output_format',output)
#            url = _append_(url,'time_range',time)
#            url = _append_(url,'level_range',level)
#            url = _append_(url,'spatial_operation',operation)
#            url = _append_(url,'aggregate',agg)
#            url = _append_(url,'calc',calc)
#            url = _append_(url,'calc_raw',calc_raw)
#            url = _append_(url,'calc_grouping',calc_grouping)
#            url = _append_(url,'s_abstraction',s_abstraction)
#            url = _append_(url,'snippet',snippet)
#            url = _append_(url,'agg_selection',agg_selection)
#
#            resp = self.c.get(url)
##            print resp.content
#            
#    def test_nc_output(self):
#        uri = '/tmp/tmp76bZWz/ocg.nc'
#        ops = OcgOperations(dataset={'uri':uri,'variable':'clt'},snippet=True,
#                            spatial_operation='intersects',aggregate=False,
#                            output_format='shp')
#        OcgInterpreter(ops).execute()
#        
#    def open_in_chrome(self,url):
#        subprocess.call(["google-chrome",'http://127.0.0.1:8000'+url])
#        
#    def test_display_inspect(self):
#        uid = '5'
#        uri = '/usr/local/climate_data/CanCM4/tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
#        
#        url1 = '/inspect?uid='+uid
#        url2 = '/inspect?uri='+uri
#        
#        variables = ['none','tasmin']
#        s_abstraction = ['point','polygon']
#        
#        for url,variable,s_abstraction in itertools.product([url1,url2],variables,s_abstraction):
#            url += '&variable={0}'.format(variable)
#            url += '&s_abstraction={0}'.format(s_abstraction)
#            resp = self.c.get(url)
#            self.assertTrue(len(resp.content) > 100)
#        
#    def test_get_shp(self):
##        url = '/shp/co_watersheds'
##        url = '/shp/world_countries'
#        url = '/shp/co_watersheds?unwrap=true&pm=0'
#
##        self.open_in_chrome(url)
#        resp = self.c.get(url)
#        
#    def test_get_snippet(self):
#        uids = ['3']
##        uids = ['/home/local/WX/ben.koziol/links/ocgis/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc']
#        variables = [
#                     'tas',
##                     'cl'
#                     ]
#        prefixes = [
##                    'none',
#                    'some_prefix'
#                    ]
#        spaces = [
#                 'none',
#                 '-123.4|45.6|-122.2|48.7',
#                 'mi_watersheds',
#                 'co_watersheds',
#                 'state_boundaries'
#                 ]
#        
#        for uid,variable,prefix,space in itertools.product(uids,variables,prefixes,spaces):
#            if len(uid) > 10:
#                url = '/snippet?uri={uid}&variable={variable}&prefix={prefix}&space={space}'
#            else:
#                url = '/snippet?uid={uid}&variable={variable}&prefix={prefix}&space={space}'
#            url = url.format(uid=uid,variable=variable,prefix=prefix,space=space)
#            resp = self.c.get(url)
##        self.open_in_chrome(url)
#
#    def test_point_abstraction(self):
#        url = '/snippet?uid=3&variable=tas&s_abstraction=point'
#        resp = self.c.get(url)
#        import ipdb;ipdb.set_trace()
#    
#    @pause
#    def test_multi_url(self):
#        url = ('/uid/1|2/variable/tasmax/level/none/time/none/space/state_boundaries'
#               '/operation/clip/aggregate/true/output/keyed')
#        resp = self.c.get(url)
#    
#    @pause
#    def test_multi_calc_url(self):
#        url = ('/uid/1|2/variable/tasmax/level/none/time/none/space/co_watersheds'
#               '/operation/clip/aggregate/true/output/meta'
#               '?calc=max_cons~max_cons_gte!threshold~305.372!operation~gte'
#               '|max_cons~max_cons_lt!threshold~273.15!operation~lte'
#               '|mean~mean_temp'
#               '|std~std_temp'
#               '|min~min_temp'
#               '|max~max_temp'
#               '&calc_grouping=month|year&calc_raw=false')
#        resp = self.c.get(url)
#    
#    @pause
#    def test_multivariate_request(self):
#        url = ('/uid/4|9/variable/tas|rhs/level/none/time/none/space/state_boundaries'
#               '/operation/intersects/aggregate/false/output/keyed'
#               '?calc=heat_index~hi!tas~tas!rhs~rhs!units~K&calc_grouping=day|month|year')
#        resp = self.c.get(url)
#    
#    def test_presentation_urls(self):
#        dataset = {'ta':'http://esg-datanode.jpl.nasa.gov/thredds/dodsC/esg_dataroot/obs4MIPs/observations/atmos/ta/mon/grid/NASA-JPL/AIRS/v20110608/ta_AIRS_L3_RetStd-v5_200209-201105.nc',
#                   'clt':'http://esg-datanode.jpl.nasa.gov/thredds/dodsC/esg_dataroot/obs4MIPs/observations/atmos/clt/mon/grid/NASA-GSFC/MODIS/v20111130/clt_MODIS_L3_C5_200003-201109.nc'}
#        
#        class PresUrl(object):
#            _urls = []
#            _ip = 'http://127.0.0.1:8000'
#            
#            def __init__(self):
#                self.c = Client()
#                
#            def run(self,url,prefix=None):
#                url_run = '{0}{1}'.format(self._ip,url)
#                if prefix is not None:
#                    url_run = '{0}&prefix={1}'.format(url_run,prefix)
#                self._urls.append(url_run)
#                print(url_run)
#                t1 = time.time()
#                ret = self.c.get(url_run)
#                t2 = time.time()
#                print('--- {0} ---\n\n'.format(t2-t1))
#                return(ret)
#        
#        pu = PresUrl()
##        env.WORKSPACE = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/presentation/20121126_esgf_f2f/output_data/2'
##        
##        ## download world and states shapefiles
##        pu.run('/shp/state_boundaries')
##        pu.run('/shp/state_boundaries?select_ugid=25&prefix=california')
##        pu.run('/shp/world_countries')
##
##        ## inspect the dataset
##        resp = pu.run('/inspect?uri={0}'.format(dataset['clt']))
###        print(resp.content)
##
##        ## get data snippet
##        pu.run('/snippet?s_abstraction=point&variable={0}&uri={1}'.format('ta',dataset['ta']),prefix='snippet_ta_point')
##        pu.run('/snippet?s_abstraction=polygon&variable={0}&uri={1}'.format('clt',dataset['clt']),prefix='snippet_clt_polygon')
##        
##        ## snippet again
##        geom_parms = [
##                      'state_boundaries',
##                      'state_boundaries&select_ugid=25',
##                      'world_countries'
##                      ]
##        geom_tags = [
##                     'states',
##                     'california',
##                     'world'
##                     ]
##        for geom_parm,geom_tag in zip(geom_parms,geom_tags):
##            url = '/snippet?variable=clt&uri={uri}&geom={geom_parm}&prefix=snippet_{geom_tag}'
##            url = url.format(uri=dataset['clt'],geom_parm=geom_parm,geom_tag=geom_tag)
##            pu.run(url)
##
#        ## subset by bounding box for area over california
#        url = '/subset?variable=clt&geom=-127.79297|32.24997|-112.98340|42.29356&prefix=bb_ca&uri={uri}'.format(uri=dataset['clt'])
#        pu.run(url)
#        url = '/subset?variable=clt&geom=-127.79297|32.24997|-112.98340|42.29356&prefix=bb_ca&spatial_operation=intersects&time_range=none&level_range=none&output_format=keyed&uri={uri}'.format(uri=dataset['clt'])
#        pu.run(url)
#        
#        ## clip aggregation/calculation query
#        url = '/subset?variable=clt&uri={uri}&time_range=2000-1-1|2000-12-31&spatial_operation=clip&aggregate=true&geom=state_boundaries&output_format=shp&calc=mean~mean_clt|min~min_clt|max~max_clt|std~std_clt&calc_raw=false&calc_grouping=year'
#        url = url.format(uri=dataset['clt'])
#        pu.run(url)
#        
#        ## meta output for aggregation/calculation query
#        url = '/subset?variable=clt&uri={uri}&time_range=2000-1-1|2000-12-31&spatial_operation=clip&aggregate=true&geom=state_boundaries&output_format=meta&calc=mean~mean_clt|min~min_clt|max~max_clt|std~std_clt&calc_raw=false&calc_grouping=year'
#        url = url.format(uri=dataset['clt'])
#        resp = pu.run(url)
##        print(resp.content)
##
#        ## netcdf
#        url = '/subset?variable=clt&uri={uri}&output_format=nc&geom=state_boundaries&prefix=clt_usa'.format(uri=dataset['clt'])
#        pu.run(url)
#        
#        import ipdb;ipdb.set_trace()
