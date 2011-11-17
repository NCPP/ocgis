from wrappers import multipolygon_operation
import datetime
import os
from osgeo import ogr
import warnings
from ocg_converter import ShpConverter
import cProfile
import pstats

import sys
sys.path.append('/home/bkoziol/Dropbox/UsefulScripts/python')
import shpitr

def filter_huron(feat):
    if feat['HUCNAME'] != 'HURON':
        return(False)
    else:
        return(True)

data = [
        dict(name='huron_huc8_watershed',
             path='/home/bkoziol/Dropbox/OpenClimateGIS/watersheds_4326.shp',
             filter=filter_huron,
             fields=['HUCNAME'])
        ]

def get_polygons(path,fields,filter):
    shp = shpitr.ShpIterator(path)
    polygons = []
    for feat in shp.iter_shapely(fields,filter=filter,as_multi=False):
        polygons.append(feat['geom'])
    return(polygons)

def f(data_kwds):
    times = 1
    polygons = get_polygons(data_kwds['path'],
                            data_kwds['fields'],
                            data_kwds['filter'])
    for ii in range(0,times):
        print(ii+1)
        sub = multipolygon_operation('http://cida.usgs.gov/qa/thredds/dodsC/maurer/monthly',
                                     'sresa1b_miroc3-2-medres_2_Prcp',
                                     ocg_opts=dict(rowbnds_name='bounds_latitude',
                                                   colbnds_name='bounds_longitude',
                                                   calendar='proleptic_gregorian',
                                                   time_units='days since 1950-01-01 00:00:0.0'),
                                     polygons=polygons,
                                     time_range=[datetime.datetime(2011,11,1),datetime.datetime(2016,12,31)],
                                     level_range=None,
                                     clip=True,
                                     union=True,
                                     in_parallel=False,
                                     max_proc=8,
                                     max_proc_per_poly=2)
        assert(sub.value.shape[2] > 0)
        shp = ShpConverter(sub,'sresa1b_miroc3-2-medres_2_Prcp')
        shp.convert(None)

for data_kwds in data:
    cProfile.run('f(data_kwds)','/tmp/foo')
    stats = pstats.Stats('/tmp/foo')
    stats.sort_stats('time')
    prev = sys.stdout
#    with open(os.path.join('/home/bkoziol/tmp',data_kwds['name']+'.csv'),'w') as out:
#        sys.stdout = out
    stats.print_stats()
#        sys.stdout.flush()
#    sys.stdout = prev
#    import ipdb;ipdb.set_trace()