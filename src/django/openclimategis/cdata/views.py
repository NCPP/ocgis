from urlparse import parse_qs
from django.http import HttpResponse
from ocgis.util.inspect import Inspect
from ocgis.util.helpers import get_temp_path
from ocgis import env
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union
import util.helpers as helpers
from ocgis.util.shp_cabinet import ShpCabinet
import os.path
from ocgis.spatial.union import union_geom_dicts


def get_data(request):
    '''The standard entry point for an OCGIS request.'''

    ops = helpers._get_operations_(request)
    ret = helpers._get_interpreter_return_(ops)
    
    if ops.output_format == 'meta':
        resp = HttpResponse(ret,content_type="text/plain")
    else:
        resp = helpers._zip_response_(ret)
    
    return(resp)
    
def display_inspect(request):
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    uri = helpers._get_uri_(query,scalar=True)
    io = Inspect(uri.value)
    report = io.__repr__()
    response = HttpResponse(report,content_type="text/plain")
    return(response)

def get_shp(request,key=None):
    sc = ShpCabinet()
    geom_dict = sc.get_geom_dict(key)
    dir_path = get_temp_path(nest=True,only_dir=True,wd=env.WORKSPACE)
    filename = '{0}.shp'.format(key)
    path = os.path.join(dir_path,filename)
    path = sc.write(geom_dict,path)
    path = os.path.split(path)[0]
    resp = helpers._zip_response_(path,filename=filename.replace('shp','zip'))
    return(resp)

def get_snippet(request):
    ops = helpers._get_operations_(request)
    if ops.geom is not None:
        ops.geom = union_geom_dicts(ops.geom)
    
    ops.level_range = 1
    ops.output_format = 'shp'
    ops.snippet = True
    ops.aggregate = False
    
    ret = helpers._get_interpreter_return_(ops)
    resp = helpers._zip_response_(os.path.split(ret)[0])
    return(resp)
