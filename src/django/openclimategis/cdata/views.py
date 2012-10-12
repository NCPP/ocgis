from urlparse import parse_qs
import datetime
from collections import OrderedDict
from shapely.geometry.polygon import Polygon
from django.http import HttpResponse
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.api.interp.interpreter import Interpreter
from ocgis.util.inspect import Inspect


class Slug(object):
    
    def __init__(self,value,default=None,scalar=False,nullable=True,
                 split_char='|'):
        self.default = default
        self.scalar = scalar
        self.nullable = nullable
        self.split_char = split_char
        self.value = self.format(str(value))
        
    def __repr__(self):
        msg = '{0}={1}'.format(self.__class__.__name__,self.value)
        return(msg)
        
    def __iter__(self):
        for ii in self.value:
            yield(ii)
        
    def format(self,value):
        value = value.lower()
        if value == 'none' and not self.nullable:
            raise(ValueError('not nullable'))
        elif value == 'none' and self.nullable:
            value = [self.default]
        else:
            value = self.split(value)
            value = [self._format_element_(v) for v in value]
            value = self._format_all_elements_(value)
        if self.scalar or value[0] is None:
            value = value[0]
        return(value)
    
    def split(self,value):
        return(value.split(self.split_char))
    
    def _format_element_(self,value):
        return(value)
    
    def _format_all_elements_(self,value):
        return(value)
    
    
class UidSlug(Slug):
    
    def __init__(self,value,query,default=None,scalar=False):
        self.query = query
        super(UidSlug,self).__init__(value,default=default,scalar=scalar)
    
    def format(self,value):
        if value.lower() == 'none':
            value = self.query['uri'][0]
        else:
            raise(NotImplementedError)
        value = value.split('|')
        if self.scalar:
            value = value[0]
        return(value)
        
        
class LevelSlug(Slug):
    
    def _format_element_(self,value):
        return(int(value))
    
    
class TimeSlug(Slug):
    
    def _format_element_(self,value):
        return(datetime.datetime.strptime(value,'%Y-%m-%d'))
    
    
class SpaceSlug(Slug):
    
    def _format_element_(self,value):
        try:
            ret = float(value)
        except ValueError:
            ret = value
        return(ret)
    
    def _format_all_elements_(self,value):
        try:
            minx,miny,maxx,maxy = value
            geom = Polygon(((minx,miny),
                           (minx,maxy),
                           (maxx,maxy),
                           (maxx,miny)))
            assert(geom.is_valid)
            ret = [{'id':1,'geom':geom}]
        except ValueError:
            sc = ShpCabinet()
            ret = sc.get_geom_dict(value[0])
        return(ret)
    
    
class BooleanSlug(Slug):
    
    def _format_element_(self,value):
        if value in ['false','f']:
            ret = False
        elif value in ['true','t']:
            ret = True
        return(ret)
    
    
class QueryParm(Slug):
    
    def __init__(self,query,key,default=None,split_char='|',scalar=False):
        self.query = query
        self.key = key
        self.default = default
        self.split_char = split_char
        self.scalar = scalar
        try:
            self.value = self.format(str(self.query[key][0]))
        except KeyError:
            self.value = default
            
    def __repr__(self):
        msg = '{0}[{1}]={2}'.format(self.__class__.__name__,self.key,self.value)
        return(msg)
    
    
class BoolQueryParm(QueryParm):
    
    def _format_element_(self,value):
        if value in ['false','f']:
            ret = False
        elif value in ['true','t']:
            ret = True
        return(ret)
    
    
class CalcQueryParm(QueryParm):
    
    def _format_element_(self,value):
        func,name = value.split('~',1)
        try:
            name,kwds_raw = name.split('!',1)
            kwds_raw = kwds_raw.split('!')
            kwds = {}
            for kwd in kwds_raw:
                kwd_name,kwd_value = kwd.split('~')
                try:
                    kwds.update({kwd_name:float(kwd_value)})
                except ValueError:
                    kwds.update({kwd_name:str(kwd_value)})
        except ValueError:
            kwds = {}
        ret = {'func':func,'name':name,'kwds':kwds}
        return(ret)


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
    ret = interp.execute()
    print(ret)
    
def display_inspect(request,uid=None):
    ## parse the query string
    query = parse_qs(request.META['QUERY_STRING'])
    uid = UidSlug(uid,query,scalar=True)
    io = Inspect(uid.value)
    report = io.__repr__()
    response = HttpResponse(report,content_type="text/plain")
    return(response)