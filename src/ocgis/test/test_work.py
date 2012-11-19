import unittest
import itertools
from collections import OrderedDict
from ocgis.api.operations import OcgOperations
from ocgis.api.iocg.interpreter_ocg import OcgInterpreter
from ocgis.util.shp_cabinet import ShpCabinet
import sys;sys.argv = ['', 'TestWork.test_get_data']
import traceback


class TestWork(unittest.TestCase):

    def test_get_data(self):
        start = 0
        for ii,ops in self.iter_operations(start=start):
            try:
                ret = OcgInterpreter(ops).execute()
            except:
                print traceback.format_exc()
                import ipdb;ipdb.set_trace()
            print(ret)

    def iter_operations(self,start=0):
        output_format = {'output_format':[
                                          'shp',
#                                          'keyed',
#                                          'nc',
                                          ]}
        snippet = {'snippet':[
                              True,
#                              False
                              ]}
        dataset = {'dataset':[
                              {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc','variable':'tasmax'},
#                              {'uri':'http://esg-datanode.jpl.nasa.gov/thredds/dodsC/esg_dataroot/obs4MIPs/observations/atmos/clt/mon/grid/NASA-GSFC/MODIS/v20111130/clt_MODIS_L3_C5_200003-201109.nc','variable':'clt'}
                              ]}
        geom = {'geom':[
                        None,
#                        self.california,
#                        self.state_boundaries
                        ]}
        aggregate = {'aggregate':[
                                  True,
#                                  False
                                  ]}
        spatial_operation = {'spatial_operation':[
#                                                  'clip',
                                                  'intersects',
                                                  ]}
        vector_wrap = {'vector_wrap':[
#                                      True,
                                      False
                                      ]}
        interface = {'interface':[
#                                  {},
                                  {'s_abstraction':'point'}
                                  ]}
        
        args = [output_format,snippet,dataset,geom,aggregate,spatial_operation,vector_wrap,interface]
        
        combined = OrderedDict()
        for arg in args: combined.update(arg)
        
        for ii,ret in enumerate(itertools.product(*combined.values())):
            if ii >= start:
                kwds = dict(zip(combined.keys(),ret))
                ops = OcgOperations(**kwds)
                yield(ii,ops)
    
    @property
    def california(self):
        sc = ShpCabinet()
        ret = sc.get_geom_dict('state_boundaries',{'id':[25]})
        return(ret)
    
    @property
    def state_boundaries(self):
        sc = ShpCabinet()
        ret = sc.get_geom_dict('state_boundaries')
        return(ret)


if __name__ == "__main__":
    unittest.main()