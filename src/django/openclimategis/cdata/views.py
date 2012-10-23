from urlparse import parse_qs
from collections import OrderedDict
from django.http import HttpResponse
from ocgis.api.interp.interpreter import Interpreter
from ocgis.util.inspect import Inspect
from util.slugs import *
from util.zipper import Zipper
from ocgis.util.helpers import get_temp_path
from ocgis import env
import os.path
from ocgis.exc import InterpreterNotRecognized
from ocgis.api.interp.iocg.interpreter_ocg import OcgInterpreter
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union


def get_data(request,uid=None,variable=None,level=None,time=None,space=None,
             operation='intersects',aggregate=True,output='keyed'):
    '''The standard entry point for an OCGIS request.'''
    
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    
    ## format the url slugs
    uid = UidSlug(uid,query)
    variable = Slug(variable)
    level = LevelSlug(level)
    time = TimeSlug(time)
    space = SpaceSlug(space)
    operation = Slug(operation,default='intersects',scalar=True)
    aggregate = BooleanSlug(aggregate,default=True,scalar=True)
    output = Slug(output,default='keyed',scalar=True)
    
    ## get the query parameters
    calc_raw = BoolQueryParm(query,'calc_raw',default=True,scalar=True)
    calc_grouping = QueryParm(query,'calc_grouping')
    calc = CalcQueryParm(query,'calc')
    backend = QueryParm(query,'backend',default='ocg')
    output_grouping = QueryParm(query,'output_grouping')
    prefix = QueryParm(query,'prefix',scalar=True)
    
    ## piece together the OCGIS operations dictionary ##########################

    ## format meta list
    meta = []
    if len(uid.value) < len(variable.value):
        for u in uid:
            for v in variable:
                meta.append({'uri':u,'variable':v})
    elif len(variable.value) < len(uid.value):
        if len(variable.value) > 1:
            raise(NotImplementedError)
        else:
            meta.append({'uri':uid.value,'variable':variable.value[0]})
    else:
        for u,v in zip(uid,variable):
            meta.append({'uri':u,'variable':v})

    ops = OrderedDict(
     meta=meta,
     time_range=time,
     level_range=level,
     spatial_operation=operation,
     aggregate=aggregate,
     output_format=output,
     calc=calc,
     calc_raw=calc_raw,
     calc_grouping=calc_grouping,
     geom=space,
     backend=backend,
     output_grouping=output_grouping
                )
    
    for key,value in ops.iteritems():
        try:
            ops.update({key:value.value})
        except AttributeError:
            ops.update({key:value})
            
    ## add request specific values
    ops['request_url'] = request.build_absolute_uri()
    ops['request_prefix'] = prefix.value
    
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
    
def display_inspect(request,uid=None):
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    uid = UidSlug(uid,query,scalar=True)
    io = Inspect(uid.value)
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
    uri = UidSlug(uid,query)
    variable = Slug(variable)
    prefix = QueryParm(query,'prefix',scalar=True)
    space = QueryParm(query,'space',scalar=True)
    
    if space.value is not None:
        space = SpaceSlug(space.value)
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
     'meta':[{'uri':uri.value[0],'variable':variable.value[0]}],
     'level_range':1,
     'output_format':'shp',
     'request_snippet':True,
     'aggregate':False,
     'request_prefix':prefix.value,
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