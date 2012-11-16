import unittest
import itertools
from collections import OrderedDict
from ocgis.api.operations import OcgOperations
from ocgis.api.iocg.interpreter_ocg import OcgInterpreter
from ocgis.util.shp_cabinet import ShpCabinet

import sys;sys.argv = ['', 'TestWork.test_coordinate_shift']
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.util.helpers import make_poly, shapely_to_shp
import time

class TestWork(unittest.TestCase):

    def test_get_data(self):
        ops = self.get_operations()
        ret = OcgInterpreter(ops).execute()
    
    def test_coordinate_shift(self):
        sc = ShpCabinet()
#        attr_filter = {'name':['Mali']}
        attr_filter = None
        geoms = sc.get_geom_dict('world_countries',attr_filter=attr_filter)
        clip1 = make_poly((-90,90),(-180,-1.40625))
        clip2 = make_poly((-90,90),(-1.40625,180))
#        clip = MultiPolygon([clip1,clip2])
        lon_cutoff = -1.40625
        
        def _get_iter_(geom):
            try:
                it = iter(geom)
            except TypeError:
                it = [geom]
            return(it)
        
        def _shift_(coords):
            return([coords[0]+360,coords[1]])
        
        def _transform_(geom,lon_cutoff):
            it = _get_iter_(geom)
            adjust = False
            for polygon in it:
                for coords in polygon.exterior.coords:
                    if any([c < lon_cutoff for c in coords]):
                        adjust = True
                        break
            if adjust:
                left = geom.intersection(clip1)
                right = geom.intersection(clip2)
                if right.is_empty:
                    right_polygons = []
                else:
                    if isinstance(right,MultiPolygon):
                        right_polygons = [poly for poly in right]
                    else:
                        right_polygons = [right]
                
#                shapely_to_shp(left,'left')
#                shapely_to_shp(right,'right')
#                tdk
#                import ipdb;ipdb.set_trace()
#                sc.write([{'geom':new_geom,'id':1}],'/tmp/spain3.shp')
#                import ipdb;ipdb.set_trace()
                if isinstance(left,Polygon):
                    left_polygons = [Polygon([_shift_(ctup) for ctup in left.exterior.coords])]
                else:
                    left_polygons = []
                    for polygon in left:
                        new_geom = Polygon([_shift_(ctup) for ctup in polygon.exterior.coords])
                        left_polygons.append(new_geom)
#                if isinstance(right,MultiPolygon):
#                    right_polygons = [poly for poly in right]
#                else:
#                    right_polygons = [right]
#                    left = MultiPolygon(polygons)
                ret = MultiPolygon(left_polygons + right_polygons)
            else:
                ret = geom
            return(ret)
        
        for geom in geoms:
            print(geom['name'])
            geom['geom'] = _transform_(geom['geom'],lon_cutoff)
            
        sc.write(geoms,'/tmp/remapped{0}.shp'.format(time.time()))
        import ipdb;ipdb.set_trace()

    def iter_operations(self):
        output_format = {'output_format':['keyed']}
        snippet = {'snippet':[
                              True,
#                              False
                              ]}
        dataset = {'dataset':[{'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc','variable':'tasmax'}]}
        
        args = [output_format,snippet,dataset]
        
        combined = OrderedDict()
        for arg in args: combined.update(arg)
        
        for ret in itertools.product(*combined.values()):
            kwds = dict(zip(combined.keys(),ret))
            ops = OcgOperations(**kwds)
            yield(ops)
            
    def get_operations(self):
        it = self.iter_operations()
        return(it.next())


if __name__ == "__main__":
    unittest.main()