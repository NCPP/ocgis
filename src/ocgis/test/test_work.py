import unittest
import itertools
from collections import OrderedDict
from ocgis.api.operations import OcgOperations
from ocgis.util.shp_cabinet import ShpCabinet
import traceback
import time
from ocgis.util.helpers import make_poly #@UnusedImport
from copy import deepcopy
from ocgis.api.interpreter import OcgInterpreter
from nose.plugins.skip import SkipTest
import shutil
import tempfile
from ocgis.exc import ExtentError
from ocgis.api.dataset.request import RequestDataset
from datetime import datetime
import os

#raise SkipTest(__name__)

class TestWork(unittest.TestCase):
    
    def _allowed_exception_(self,ops,e):
        ret = False
        ## point-based abstractions will result in empty state geometries
        if ops.allow_empty is False and type(e) == ExtentError and ops.abstraction == 'point':
            if len(ops.geom) == 51:
                ret = True
        ## computations with netcdf output is currently not supported
        elif ops.output_format == 'nc' and ops.calc is not None and type(e) == NotImplementedError:
            ret = True
        return(ret)

    def test_get_data(self):
        start = 0
                
        for ii,ops in self.iter_operations(start=start):
            print(ii)
            ret = None
            
            try:
                ret = OcgInterpreter(ops).execute()
            except Exception as e:
                if self._allowed_exception_(ops,e) is False:
                    print traceback.format_exc()
                    import ipdb;ipdb.set_trace()
            finally:
                if ret is not None and ret.startswith(tempfile.gettempdir()):
                    print(ret)
                    if any([ret.endswith(ext) for ext in ['shp','csv','nc']]):
                        ret = os.path.split(ret)[0]
                    shutil.rmtree(ret)
                    
    def iter_operations(self,start=0):
        datasets = {1:{'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                       'variable':'tasmax',
                       'alias':'tasmax'},
                    2:{'uri':'/usr/local/climate_data/CanCM4/tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                       'variable':'tasmin',
                       'alias':'tasmin'}}
        
        output_format = {'output_format':[
                                          'shp',
                                          'keyed',
                                          'meta',
                                          'nc',
                                          'csv'
                                          ]}
        snippet = {'snippet':[
                              True,
#                              False
                              ]}
        
        dataset = {'dataset':[
                              [1],
                              [1,2]
                              ]}
                   
        geom = {'geom':[
                        self.alaska,
                        None,
                        self.california,
                        self.state_boundaries,
                        [{'ugid':1,'geom':make_poly((24.2,50.8),(-128.7,-65.2))}],
#                        self.world_countries
                        ]}
        aggregate = {'aggregate':[
                                  True,
                                  False
                                  ]}
        spatial_operation = {'spatial_operation':[
                                                  'clip',
                                                  'intersects',
                                                  ]}
        vector_wrap = {'vector_wrap':[
                                      True,
                                      False
                                      ]}
        abstraction = {'abstraction':[
                                      'polygon',
                                      'point'
                                      ]}
        
        agg_selection = {'agg_selection':[
                                          True,
                                          False
                                          ]}
        
        level_range = {'level_range':[
                                      None,
                                      [1,1]
                                      ]}
        time_range = {'time_range':[
                                    [datetime(2001,1,1),datetime(2001,12,31,23,59,59)],
                                    None
                                    ]}
        allow_empty = {'allow_empty':[
                                      True,
                                      False
                                      ]}
        calc = {'calc':[
                        [{'func':'mean','name':'my_mean'}],
                        None,
                        ]}
        calc_grouping = {'calc_grouping':[
                                          ['month','year'],
                                          ['year']
                                        ]
                         }
        
        args = [output_format,snippet,dataset,geom,aggregate,spatial_operation,
                vector_wrap,abstraction,agg_selection,level_range,time_range,
                allow_empty,calc,calc_grouping]
        
        combined = OrderedDict()
        for arg in args: combined.update(arg)
        
        for ii,ret in enumerate(itertools.product(*combined.values())):
            if ii >= start:
                kwds = deepcopy(dict(zip(combined.keys(),ret)))
                time_range = kwds.pop('time_range')
                level_range = kwds.pop('level_range')
                rds = [RequestDataset(datasets[jj]['uri'],
                                      datasets[jj]['variable'],
                                      time_range=time_range,
                                      level_range=level_range)
                       for jj in kwds['dataset']]
                kwds['dataset'] = rds
                ops = OcgOperations(**kwds)
                yield(ii,ops)
    
    @property
    def california(self):
        sc = ShpCabinet()
        ret = sc.get_geom_dict('state_boundaries',{'ugid':[25]})
        return(ret)
    
    @property
    def alaska(self):
        sc = ShpCabinet()
        ret = sc.get_geom_dict('state_boundaries',{'ugid':[51]})
        return(ret)
    
    @property
    def state_boundaries(self):
        sc = ShpCabinet()
        ret = sc.get_geom_dict('state_boundaries')
        return(ret)
    
    @property
    def world_countries(self):
        sc = ShpCabinet()
        ret = sc.get_geom_dict('world_countries')
        return(ret)

    def test_profile(self):
        raise(SkipTest)
        prev = sys.stdout
        with open('/tmp/out.txt','w') as f:
            sys.stdout = f
            start = 0
            for ii,ops in self.iter_operations(start=start):
                t1 = time.time()
                OcgInterpreter(ops).execute()
                t2 = time.time()
                if int(ops.geom[0]['geom'].area) == 1096:
                    geom = 'states'
                else:
                    geom = 'bb'
                prnt = [geom,ops.dataset[0]['uri'],ops.output_format,t2-t1]
                print ','.join(map(str,prnt))
                time.sleep(5)
#                break
        sys.stdout = prev


if __name__ == "__main__":
    unittest.main()
