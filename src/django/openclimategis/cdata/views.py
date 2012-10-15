from urlparse import parse_qs
from collections import OrderedDict
from django.http import HttpResponse
from ocgis.api.interp.interpreter import Interpreter
from ocgis.util.inspect import Inspect
from util.slugs import *
from util.zipper import Zipper


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
    
    interp = Interpreter.get_interpreter(ops)
    path = interp.execute()
    
    zipper = Zipper(path)
    zip_stream = zipper.get_zip_stream()
    
    import ipdb;ipdb.set_trace()
    print(ret)
    
def display_inspect(request,uid=None):
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    uid = UidSlug(uid,query,scalar=True)
    io = Inspect(uid.value)
    report = io.__repr__()
    response = HttpResponse(report,content_type="text/plain")
    return(response)