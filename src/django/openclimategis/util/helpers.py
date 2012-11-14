from util.zipper import Zipper
import datetime
from ocgis import env
from django.http import HttpResponse
from urlparse import parse_qs
import util.parms as parms
import ocgis.meta.interface.models as imodels
import exc
from ocgis.exc import InterpreterNotRecognized
from ocgis.api.interpreter import Interpreter
from ocgis.api.iocg.interpreter_ocg import OcgInterpreter
from ocgis.api.operations import OcgOperations
from ocgis.api.definition import OcgParameter


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
        uri = parms.UidParm(query,'uid',scalar=scalar)
    except exc.QueryParmError:
        uri = parms.UriParm(query,'uri',scalar=scalar)
    return(uri)

def _get_interface_overload_(query):
    mmap = {'s_row':imodels.Row,
            's_column':imodels.Column,
            's_row_bounds':imodels.RowBounds,
            's_column_bounds':imodels.ColumnBounds,
            's_proj':None,
            's_abstraction':None,
            't_calendar':imodels.Calendar,
            't_units':imodels.TimeUnits,
            't_variable':imodels.Time,
            'l_variable':imodels.Level}
    
    name_map = {}
    
    for key,value in mmap.iteritems():
        qp = parms.QueryParm(query,key,scalar=True)
        if value is None:
            value = key
        name_map.update({value:qp.value})
        
    return(name_map)
    
def _get_interpreter_return_(ops):
    try:
        interp = Interpreter.get_interpreter(ops)
    except InterpreterNotRecognized:
        interp = OcgInterpreter(ops)
    ret = interp.execute()
    return(ret)

def _get_operations_(request):
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    
    ## get dataset information
    uri = _get_uri_(query)
    variable = parms.OcgQueryParm(query,'variable',nullable=False)
    dataset = []
    if len(uri.value) < len(variable.value):
        for u in uri:
            for v in variable:
                dataset.append({'uri':u,'variable':v})
    elif len(variable.value) < len(uri.value):
        if len(variable.value) > 1:
            raise(NotImplementedError)
        else:
            dataset.append({'uri':uri.value,'variable':variable.value[0]})
    else:
        for u,v in zip(uri,variable):
            dataset.append({'uri':u,'variable':v})
    
    ## initialize initial operations object
    ops = OcgOperations(dataset=dataset)
    
    ## iterate objects parsing the query dictionary
    for value in ops.__dict__.itervalues():
        if isinstance(value,OcgParameter) and value.name != 'dataset':
            value.parse_query(query)
            
    ## pull interface overload information
    ops.interface = _get_interface_overload_(query)
    
    ## add request specific values
    ops.request_url = request.build_absolute_uri()

    return(ops)