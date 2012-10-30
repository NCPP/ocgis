from urlparse import parse_qs
from collections import OrderedDict
from django.http import HttpResponse
from ocgis.api.interp.interpreter import Interpreter
from ocgis.util.inspect import Inspect
from util.parms import *
from util.zipper import Zipper
from ocgis.util.helpers import get_temp_path
from ocgis import env
from ocgis.exc import InterpreterNotRecognized
from ocgis.api.interp.iocg.interpreter_ocg import OcgInterpreter
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union


def get_data(request):
    '''The standard entry point for an OCGIS request.'''
    
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    
    ## extract from request
    uri = _get_uri_(query)
    variable = OcgQueryParm(query,'variable',nullable=False)
    level = LevelParm(query,'level_range')
    time = TimeParm(query,'time_range')
    space = SpaceParm(query,'geom',nullable=False)
    operation = OcgQueryParm(query,'spatial_operation',default='intersects',scalar=True)
    aggregate = BooleanParm(query,'aggregate',default=True,scalar=True)
    output = OcgQueryParm(query,'output_format',default='keyed',scalar=True)
    calc_raw = BooleanParm(query,'calc_raw',default=True,scalar=True)
    calc_grouping = OcgQueryParm(query,'calc_grouping')
    calc = CalcParm(query,'calc')
    backend = QueryParm(query,'backend',default='ocg')
    output_grouping = QueryParm(query,'output_grouping')
    prefix = QueryParm(query,'request_prefix',scalar=True)

    ## piece together the OCGIS operations dictionary ##########################

    ## format meta list
    meta = []
    if len(uri.value) < len(variable.value):
        for u in uri:
            for v in variable:
                meta.append({'uri':u,'variable':v})
    elif len(variable.value) < len(uri.value):
        if len(variable.value) > 1:
            raise(NotImplementedError)
        else:
            meta.append({'uri':uri.value,'variable':variable.value[0]})
    else:
        for u,v in zip(uri,variable):
            meta.append({'uri':u,'variable':v})

    ## construct operations dictionary
    items = [level,time,space,operation,aggregate,output,calc_raw,
             calc_grouping,calc,backend,output_grouping,prefix]
    ops = dict([[ii.key,ii.value] for ii in items])
    ops.update({'meta':meta})
    
#    import ipdb;ipdb.set_trace()
#    ops = OrderedDict(
#     meta=meta,
#     time_range=time,
#     level_range=level,
#     spatial_operation=operation,
#     aggregate=aggregate,
#     output_format=output,
#     calc=calc,
#     calc_raw=calc_raw,
#     calc_grouping=calc_grouping,
#     geom=space,
#     backend=backend,
#     output_grouping=output_grouping
#                )
#    
#    for key,value in ops.iteritems():
#        try:
#            ops.update({key:value.value})
#        except AttributeError:
#            ops.update({key:value})
            
    ## add request specific values
    ops['request_url'] = request.build_absolute_uri()
#    ops['request_prefix'] = prefix.value
    ret = _get_interpreter_return_(ops)
    
    if output.value == 'meta':
        resp = HttpResponse(ret,content_type="text/plain")
    else:
        resp = _zip_response_(ret)
    
    return(resp)

def _get_interpreter_return_(ops):
    try:
        interp = Interpreter.get_interpreter(ops)
    except InterpreterNotRecognized:
        interp = OcgInterpreter(ops)
    ret = interp.execute()
    return(ret)
    
def display_inspect(request):
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    uri = _get_uri_(query,scalar=True)
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
    resp = _zip_response_(path,filename=filename.replace('shp','zip'))
    return(resp)

def get_snippet(request,uid=None,variable=None):
    query = _get_query_dict_(request)
    uri = _get_uri_(query,scalar=True)
    variable = OcgQueryParm(query,'variable',scalar=True,nullable=False)
    prefix = OcgQueryParm(query,'request_prefix',scalar=True)
    space = SpaceParm(query,'geom')

    if space.value is not None:
        if len(space.value) > 1:
            ugeom = []
            for dct in space.value:
                geom = dct['geom']
                if isinstance(geom,MultiPolygon):
                    for poly in geom:
                        ugeom.append(poly)
                else:
                    ugeom.append(geom)
            ugeom = cascaded_union(ugeom)
            space.value = {'id':1,'geom':ugeom}
    
    ops = {
     'meta':[{'uri':uri.value,'variable':variable.value}],
     'level_range':1,
     'output_format':'shp',
     'request_snippet':True,
     'aggregate':False,
     prefix.key:prefix.value,
     'geom':space.value
           }
    
    ret = _get_interpreter_return_(ops)
    resp = _zip_response_(os.path.split(ret)[0])
    return(resp)
    
def _zip_response_(path,filename=None):
    zip_stream = Zipper(path).get_zip_stream()
    if filename is None:
        dt = str(datetime.datetime.utcnow())
        dt = dt.replace('-','')
        dt = dt.replace(' ','_')
        dt = dt.split('.')[0]
        dt = dt.replace(':','')
        filename = '{1}_{0}.zip'.format(dt,env.BASE_NAME)
    resp = HttpResponse(zip_stream,mimetype='application/zip')
    resp['Content-Disposition'] = 'attachment; filename={0}'.format(filename)
    resp['Content-length'] = str(len(zip_stream))
    return(resp)

def _get_query_dict_(request):
    query = parse_qs(request.META['QUERY_STRING'])
    return(query)

def _get_uri_(query,scalar=False):
    try:
        uri = UidParm(query,'uid',scalar=scalar)
    except exc.QueryParmError:
        uri = UriParm(query,'uri',scalar=scalar)
    return(uri)