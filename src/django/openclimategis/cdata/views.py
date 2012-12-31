from urlparse import parse_qs
from django.http import HttpResponse
from ocgis.util.inspect import Inspect
from ocgis.util.helpers import get_temp_path, union_geoms
from ocgis import env
import util.helpers as helpers
from ocgis.util.shp_cabinet import ShpCabinet
import os.path
from ocgis.api.definition import SelectUgid, Prefix, Unwrap, PrimeMeridian
from ocgis.util.spatial.wrap import unwrap_geoms
from util.parms import QueryParm
from util.helpers import _get_interface_overload_


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
    variable = QueryParm(query,'variable',scalar=True)
    interface_overload = _get_interface_overload_(query)
    io = Inspect(uri.value,variable=variable.value,
                 interface_overload=interface_overload)
    report = io.__repr__()
    response = HttpResponse(report,content_type="text/plain")
    return(response)

def get_shp(request,key=None):
    query = helpers.parse_qs(request.META['QUERY_STRING'])
    
    select_ugid = SelectUgid()
    select_ugid.parse_query(query)
    
    prefix = Prefix()
    prefix.parse_query(query)
    
    unwrap = Unwrap()
    unwrap.parse_query(query)
    
    pm = PrimeMeridian()
    pm.parse_query(query)
    
    sc = ShpCabinet()
    geom_dict = sc.get_geom_dict(key,attr_filter=select_ugid.value)
    
    ## unwrap coordinates if requested
    if unwrap.value:
        unwrap_geoms(geom_dict,pm.value)
    
    dir_path = get_temp_path(nest=True,only_dir=True,wd=env.WORKSPACE)
    if prefix.value is None:
        out_name = key
    else:
        out_name = prefix.value
    filename = '{0}.shp'.format(out_name)
    path = os.path.join(dir_path,filename)
    path = sc.write(geom_dict,path)
    path = os.path.split(path)[0]
    
    resp = helpers._zip_response_(path,filename=filename.replace('shp','zip'))
    return(resp)

def get_snippet(request):
    ops = helpers._get_operations_(request)
    if ops.geom is not None:
        if ops.select_ugid is not None:
            geom = ops._get_object_('geom')
            geom._filter_by_ugid_(ops.select_ugid['ugid'])
            ops.select_ugid = None
        ops.geom = union_geoms(ops.geom)
    
    ops.level_range = 1
    ops.output_format = 'shp'
    ops.snippet = True
    ops.aggregate = False
    ops.spatial_operation = 'intersects'

    ret = helpers._get_interpreter_return_(ops)
    resp = helpers._zip_response_(os.path.split(ret)[0])
    return(resp)
