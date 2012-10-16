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
    
    ## piece together the OCGIS operations dictionary ##########################

    ## format meta list
    meta = []
    if len(uid.value) < len(variable.value):
        for u in uid:
            for v in variable:
                meta.append({'uri':u,'variable':v})
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
            
    ## add request url for the meta handler
    ops['request_url'] = request.build_absolute_uri()
    
    interp = Interpreter.get_interpreter(ops)
    ret = interp.execute()
    
    if output.value == 'meta':
        resp = HttpResponse(ret,content_type="text/plain")
    else:
        zipper = Zipper(ret)
        zip_stream = zipper.get_zip_stream()
        resp = _zip_response_(zip_stream)
    
    return(resp)
    
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
    path = os.path.join(dir_path,'{0}.shp'.format(key))
    path = sc.write(geom_dict,path)
    zipper = Zipper(os.path.split(path)[0])
    zip_stream = zipper.get_zip_stream()
    resp = _zip_response_(zip_stream)
    return(resp)
    
def _zip_response_(zip_stream,filename=None):
    if filename is None:
        dt = str(datetime.datetime.utcnow())
        dt = dt.replace('-','')
        dt = dt.replace(' ','_')
        dt = dt.split('.')[0]
        dt = dt.replace(':','')
        filename = 'ocg_{0}.zip'.format(dt)
    resp = HttpResponse(zip_stream,mimetype='application/zip')
    resp['Content-Disposition'] = 'attachment; filename={0}'.format(filename)
    resp['Content-length'] = str(len(zip_stream))
    return(resp)