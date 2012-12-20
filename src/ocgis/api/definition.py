from datetime import datetime
from copy import copy, deepcopy
from types import NoneType
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.calc.base import OcgFunctionTree, OcgCvArgFunction
from ocgis.calc import library
import numpy as np


class OcgParameter(object):
    '''
    >>> op = OcgParameter('aggregate',bool,nullable=True,default=True)
    >>> op.value = False
    >>> op = OcgParameter('output_format',str,nullable=False,default='keyed')
    >>> op.value
    'keyed'
    >>> op.value = None; op.value
    'keyed'
    >>> op.value = 'shp'
    >>> op.value
    'shp'
    >>> query = {'output_format':['nc']}
    >>> op.parse_query(query)
    >>> op.value
    'nc'
    >>> op.parse_query({'output_format':[5]})
    Traceback (most recent call last):
    ...
    AttributeError: 'int' object has no attribute 'lower'
    >>> op = OcgParameter('foo',str).parse_query(query)
    Traceback (most recent call last):
    ...
    ValueError: "foo" is not nullable.
    '''
    
    def __init__(self,name,dtype,nullable=False,default=None,length=None,
                 alias=None,init_value=None):
        self.name = name
        self.nullable = nullable
        self.default = default
        self.dtype = dtype
        self.length = length
        self.alias = alias
        
        self._first_set = True
        self._value = None
        if init_value is not None:
            self.value = init_value
       
    @property
    def value(self):
        return(self.format(self._value))
    @value.setter
    def value(self,value):
        if self._first_set:
            value = deepcopy(value)
            self._first_set = False
        self._value = self.format(value)
        
    def format(self,value):
        if value is None:
            if self.default is None and self.nullable is False:
                raise(ValueError('"{0}" is not nullable.'.format(self.name)))
            else:
                ret = self.default
        else:
            ret = self._format_(value)
            self._assert_(type(ret) == self.dtype)
            if self.length is not None:
                self._assert_(len(ret) == self.length)
            self.validate(ret)
        return(ret)
    
    def format_string(self,value):
        return(self._format_string_(value.lower()))
    
    def parse_query(self,query):
        try:
            value = query[self.name]
        except KeyError:
            value = query.get(self.alias)
        if value is not None:
            value = value[0]
            try:
                self.value = self.format_string(value)
            except:
                if value == 'none':
                    self.value = None
                else:
                    raise
        else:
            self.value = None
        
    def validate(self,value):
        pass
    
    def message(self):
        raise(NotImplementedError)
    
    def _assert_(self,test,msg=None):
        assert(test)
    
    def _format_(self,value):
        return(value)
    
    def _format_string_(self,value):
        return(value)
    
    
class BooleanParameter(OcgParameter):
    _dtype = bool
    
    def format_string(self,value):
        if value in ['t','true','1']:
            ret = True
        elif value in ['f','false','0']:
            ret = False
        return(ret)
    

class AttributedOcgParameter(OcgParameter):
    _name = None
    _dtype = None
    _nullable = False
    _default = None
    _alias = None
    _length = None
    
    def __init__(self,init_value=None):
        super(AttributedOcgParameter,self).__init__(
         self._name,self._dtype,nullable=self._nullable,default=self._default,
         length=self._length,alias=self._alias,init_value=init_value)
        
        
class TimeRange(AttributedOcgParameter):
    '''
    >>> time_range = TimeRange()
    >>> time_range.value = 4
    Traceback (most recent call last):
    ....
    AssertionError
    >>> time_range.value = [datetime(2000,1,1),datetime(1999,1,1)]
    Traceback (most recent call last):
    ...
    AssertionError
    >>> time_range.parse_query({'time_range':['2000-1-1|1999-1-1']})
    Traceback (most recent call last):
    ...
    AssertionError
    >>> time_range.parse_query({'time_range':['2000-1-1|2000-1-1']})
    >>> time_range.value
    [datetime.datetime(2000, 1, 1, 0, 0), datetime.datetime(2000, 1, 1, 23, 59, 59)]
    >>> time_range.message()
    "Inclusive time selection range is: ['2000-01-01 00:00:00', '2000-01-01 23:59:59']"
    >>> time_range.parse_query({'time_range':['2000-1-1|2000-100-1']})
    Traceback (most recent call last):
    ...
    ValueError: time data '2000-100-1' does not match format '%Y-%m-%d'
    >>> time_range.parse_query({'time_range':['none']})
    >>> time_range.value
    >>> time_range.value = None
    >>> time_range.message()
    'All time points returned.'
    '''
    _name = 'time_range'
    _dtype = list
    _nullable = True
    _length = None
    
    def validate(self,value):
        for v in value:
            self._assert_(v[0] <= v[1])
            
    def _format_(self,value):
        if type(value[0]) == datetime:
            ret = [value]
        else:
            ret = value
        return(ret)
        
    def _format_string_(self,value):
        ret = [datetime.strptime(v,'%Y-%m-%d') for v in value.split('|')]
        ## ensure the time range is inclusive
        d = ret[1]
        ret[1] = datetime(d.year,d.month,d.day,23,59,59)
        return([ret])
    
    def message(self):
        if self.value is None:
            msg = 'All time points returned.'
        else:
            msg = 'Inclusive time selection range is: {0}'.format([str(v) for v in self.value])
        return(msg)
    
    
class RequestUrl(AttributedOcgParameter):
    _name = 'request_url'
    _nullable = True
    _dtype = str
    
    def message(self):
        msg = 'Requested URL:\n{0}'.format(self.value)
        return(msg)
    
    
class Snippet(BooleanParameter,AttributedOcgParameter):
    '''
    >>> snippet = Snippet(); snippet.value
    False
    >>> snippet.value = snippet.format_string('true'); snippet.value
    True
    '''
    _name = 'snippet'
    _nullable = True
    _default = False
    
    def message(self):
        if self.value:
            msg = 'A data snippet was returned. Only the fist time point/range and first level were selected.'
        else:
            msg = 'All data returned.'
        return(msg)


class Prefix(AttributedOcgParameter):
    '''
    >>> p = Prefix(init_value='foo')
    >>> p.value
    'foo'
    >>> p = Prefix(init_value=5)
    Traceback (most recent call last):
    ...
    AssertionError
    '''
    _dtype = str
    _name = 'prefix'
    _nullable = True
    
    def message(self):
        msg = 'Data output given the following user-defined prefix: {0}'.format(self.value)
        return(msg)

    
class AggregateSelection(BooleanParameter,AttributedOcgParameter):
    _name = 'agg_selection'
    _nullable = True
    _default = False
    
    def message(self):
        if self.value:
            msg = 'Selection geometries were aggregated (unioned).'
        else:
            msg = 'Selection geometries left as is.'
        return(msg)


class Backend(AttributedOcgParameter):
    _name = 'backend'
    _nullable = True
    _default = 'ocg'
    _dtype = str
    
    def validate(self,value):
        self._assert_(value in ['ocg'])
    
    def message(self):
        if self.value == 'ocg':
            msg = ('OpenClimateGIS used as geoprocessing and calculation '
                   'backend.')
        else:
            raise(NotImplementedError)
        return(msg)
    
    
class CalcGrouping(AttributedOcgParameter):
    '''
    >>> cg = CalcGrouping(); cg.value
    ['day', 'month', 'year']
    >>> cg.value = 'forever'
    Traceback (most recent call last):
    ...
    AssertionError
    >>> cg.value = cg.format_string('day|month'); cg.value
    ['day', 'month']
    >>> cg.value = cg.format_string('day|forever'); cg.value
    Traceback (most recent call last):
    ...
    AssertionError
    '''
    _name = 'calc_grouping'
    _nullable = True
    _default = ['day','month','year']
    _dtype = list
    
    def validate(self,value):
        for ii in value:
            self._assert_(ii in ['day','month','year'])
            
    def _format_(self,value):
        return(list(set(value)))
    
    def _format_string_(self,value):
        grouping = value.split('|')
        return(list(grouping))
    
    def message(self):
        msg = ('Temporal aggregation determined by the following group(s): {0}')
        msg = msg.format(self.value)
        return(msg)
    
    
class LevelRange(AttributedOcgParameter):
    '''
    >>> level_range = LevelRange()
    >>> level_range.value = 1; level_range.value
    [1, 1]
    '''
    _name = 'level_range'
    _default = [[1,1]]
    _dtype = list
    _nullable = True
    _length = None
    
    def _format_(self,value):
        try:
            v = int(value)
            ret = [[v,v]]
        except TypeError:
            ret = value
        ret = np.array(ret,dtype=int).tolist()
        return(ret)
    
    def _format_string_(self,value):
        values = [int(ii) for ii in value.split('|')]
        return(values)
    
    def validate(self,value):
        for v in value:
            self._assert_(v[0] <= v[1])

    def message(self):
        if self.value is None:
            msg = ('No level range provided. If variable(s) has/have a level dimesion,'
                   ' all levels will be returned.')
        else:
            msg = 'Inclusive level range returned is: {0}.'.format(self.value)
        return(msg)
    
    
class OutputFormat(AttributedOcgParameter):
    _possible = ['numpy','shpidx','shp','csv','nc','keyed','meta']
    _name = 'output_format'
    _nullable = True
    _default = 'keyed'
    _dtype = str
    
    def message(self):
        mmap = {'numpy':'an OpenClimateGIS data format storing variables as NumPy arrays.',
                'shpidx':'an ESRI Shapefile mapping unique geometry identifiers (GID) to geometries constructed from the dataset.',
                'shp':'an ESRI Shapefile containing all output data.',
                'csv':'a Comma Separated Value files containing all output data.',
                'keyed':'a group of keyed files linking to the "value" file. Files are named after the unique key for which they contain data. A geometry index is also included.',
                'meta':'a description of the operations performed or to be performed given provided parameters. No data is touched during the operation. Hence, operations may fail if dataset and/or parameters are somehow noncompliant.'
                }
        msg = 'Output format is "{0}" which is {1}'.format(self.value,mmap[self.value])
        return(msg)
    
    def validate(self,value):
        self._assert_(value in self._possible)
        
        
class SpatialOperation(AttributedOcgParameter):
    _possible = ['intersects','clip']
    _name = 'spatial_operation'
    _nullable = True
    _default = 'intersects'
    _dtype = str
    
    def message(self):
        if self.value == 'intersects':
            msg = 'The "intersects" operation returns all grid cells overlapping the selection geometry. Note that this does NOT include those geometries touching the selection geometry. In the case of point data, only those points occurring inside the selection geometry are returned.'
        if self.value == 'clip':
            msg = 'The "clip" operation is a full geometric intersection of the selection and target geometries.'
        return(msg)
    
    def validate(self,value):
        self._assert_(value in self._possible)
        
        
class CalcRaw(BooleanParameter,AttributedOcgParameter):
    _name = 'calc_raw'
    _nullable = True
    _default = False
    
    def message(self):
        if self.value:
            msg = 'Raw values will be used for calculations. These are the original data values linked to a selection value.'
        else:
            msg = 'Aggregated values will be used during the calculation.'
        return(msg)
    
    
class Aggregate(BooleanParameter,AttributedOcgParameter):
    _name = 'aggregate'
    _nullable = True
    _default = False
    
    def message(self):
        if self.value:
            msg = 'Selected geometries are aggregated (unioned), and associated data values are area-weighted based on final area following the spatial operation. Weights are normalized using the maximum area of the geometry set.'
        else:
            msg = 'Selected geometries are not aggregated (unioned).'
        return(msg)
    
    
class Calc(AttributedOcgParameter):
    '''
    >>> calc = Calc()
    >>> calc.value = {'func':'n','name':'some_n'}; calc.value
    [{'ref': <class 'ocgis.calc.wrap.library.SampleSize'>, 'name': 'some_n', 'func': 'n', 'kwds': {}}]
    >>> calc.value = {'func':'mean'}
    Traceback (most recent call last):
    ...
    AssertionError
    >>> calc.value = [{'func':'mean','name':'my_mean'}]; calc.value
    [{'ref': <class 'ocgis.calc.wrap.library.SampleSize'>, 'name': 'n', 'func': 'n', 'kwds': {}}, {'ref': <class 'ocgis.calc.wrap.library.Mean'>, 'name': 'my_mean', 'func': 'mean', 'kwds': {}}]
    >>> calc.format_string('mean~my_mean|max~my_max')
    [{'name': 'my_mean', 'func': 'mean', 'kwds': {}}, {'name': 'my_max', 'func': 'max', 'kwds': {}}]
    >>> calc.value = calc.format_string('min~my_min|between~btw5_10!lower~5!upper~10')
    >>> calc.value
    [{'ref': <class 'ocgis.calc.wrap.library.SampleSize'>, 'name': 'n', 'func': 'n', 'kwds': {}}, {'ref': <class 'ocgis.calc.wrap.library.Min'>, 'name': 'my_min', 'func': 'min', 'kwds': {}}, {'ref': <class 'ocgis.calc.wrap.library.Between'>, 'name': 'btw5_10', 'func': 'between', 'kwds': {'upper': 10.0, 'lower': 5.0}}]
    '''
    _name = 'calc'
    _nullable = True
    _dtype = list
    
    def _format_(self,value):
        if type(value) not in [list,tuple]:
            value = [value]
        funcs_copy = copy(value)
        if not any([ii['func'] == 'n' for ii in value]):
            funcs_copy.insert(0,{'func':'n'})
        
        potentials = OcgFunctionTree.get_potentials()
        for f in funcs_copy:
            for p in potentials:
                if p[0] == f['func']:
                    f['ref'] = getattr(library,p[1])
                    break
            if 'name' not in f and f['func'] == 'n':
                f['name'] = f['func']
            if 'kwds' not in f:
                f['kwds'] = {}
            else:
                for key,value in f['kwds'].iteritems():
                    try:
                        f['kwds'][key] = value.lower()
                    except AttributeError:
                        pass
        return(funcs_copy)
    
    def _format_string_(self,value):
        ret = []
        funcs = value.split('|')
        for func in funcs:
            key,uname = func.split('~',1)
            try:
                uname,kwds_raw = uname.split('!',1)
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
            dct = {'func':key,'name':uname,'kwds':kwds}
            ret.append(dct)
        return(ret)
    
    def validate(self,value):
        for ii in value:
            self._assert_('func' in ii,('at least the function name is '
                                        'required using the key "func"'))
            if ii['func'] != 'n':
                self._assert_('name' in ii,'a custom function name is required.')
            
    def message(self):
        if self.value is None:
            msg = 'No calculations requested.'
        else:
            msg = ''
            for ii in self.value:
                msg += '* {0} :: {1}\n'.format(ii['name'],ii['ref'].description)
                if len(ii['kwds']) > 0:
                    msg += '** Parameters:\n'
                    for key,value in ii['kwds'].iteritems():
                        msg += '*** {0}={1}\n'.format(key,value)
                msg += '\n'
        return(msg[:-2])
    
    
class Geom(AttributedOcgParameter):
    '''
    >>> geom = Geom()
    >>> formatted = geom.format_string('-123.4|45.67|-156.5|48.25')
    >>> formatted[0]['geom'].bounds
    (-156.5, 45.67, -123.4, 48.25)
    '''
    _name = 'geom'
    _nullable = True
    _default = [{'geom':None,'ugid':1}]
    _dtype = list
    
    def _format_(self,value):
#        value = deepcopy(value)
        if type(value) not in [list,tuple]:
            value = [value]
        return(value)
    
    def validate(self,value):
        for ii in value:
            self._assert_(type(ii) == dict,
             'list elements must be dictionaries with keys "ugid" and "geom"')
            self._assert_('ugid' in ii,'a geom must have a "ugid" key')
            self._assert_('geom' in ii,'a geom dict must have a "geom" key')
            self._assert_(type(ii['geom']) in [NoneType,Polygon,MultiPolygon],
                          'geometry type not recognized')
    
    def message(self):
        for ii in self.value:
            if ii['geom'] is None:
                msg = 'No user-supplied geometry. All data returned.'
                return(msg)
            else:
                msg = '{0} user geometries provided.'.format(len(self.value))
                return(msg)
            
    def _format_string_(self,value):
        elements = value.split('|')
        try:
            elements = [float(e) for e in elements]
            minx,miny,maxx,maxy = elements
            geom = Polygon(((minx,miny),
                            (minx,maxy),
                            (maxx,maxy),
                            (maxx,miny)))
            self._assert_(geom.is_valid)
            ret = [{'id':1,'geom':geom}]
        except ValueError:
            sc = ShpCabinet()
            ret = sc.get_geom_dict(value)
        return(ret)
    
    def _filter_by_ugid_(self,ugids):
        def _filter_(geom_dict):
            if geom_dict['id'] in ugids:
                return(True)
        self.value = filter(_filter_,self.value)
    
    @property
    def is_empty(self):
        if self.value[0]['geom'] is None:
            ret = True
        else:
            ret = False
        return(ret)
    
    
class Dataset(AttributedOcgParameter):
    _name = 'dataset'
    _nullable = False
    _dtype = list
    
    def _format_(self,value):
#        value = deepcopy(value)
        if type(value) not in [list,tuple]:
            value = [value]
        return(value)
    
    def validate(self,value):
        for ii in value:
            self._assert_(type(ii) == dict,'meta list elements must be dicts')
            self._assert_('uri' in ii,'a uri must be provided')
            self._assert_('variable' in ii,'a variable must be provided')
    
    def message(self):
        lines = []
        for ii in self.value:
            lines.append('* Variable "{0}" requested from dataset with URI "{1}".'.format(ii['variable'],ii['uri']))
        return('\n'.join(lines))
    
    
class Interface(AttributedOcgParameter):
    _name = 'interface'
    _nullable = True
    _dtype = dict
    _default = {}
    
    def validate(self,value):
        for key,val in value.iteritems():
#            try:
#                assert(issubclass(key,Element))
#            except (TypeError,AssertionError):
#                self._assert_(key in ['s_proj','s_abstraction'],'interface key not a subclass of "Element"')
            if val is not None:
                self._assert_(type(val) == str,'interface values must be strings')
    
    def message(self):
        msg = ['Interface parameter arguments:']
        for key,value in self.value.iteritems():
            try:
                name = key._iname
            except AttributeError:
                name = key
            msg2 = ' {0} :: {1}'.format(name,value)
            msg.append(msg2)
        msg = '\n'.join(msg)
        return(msg)
    
    
class SelectUgid(AttributedOcgParameter):
    _name = 'select_ugid'
    _nullable = True
    _default = None
    _dtype = dict
    
    def _format_string_(self,value):
        elements = value.split('|')
        elements = [int(ii) for ii in elements]
        return({'ugid':elements})
    
    def message(self):
        if self.value is None:
            msg = 'All user geometries returned.'
        else:
            msg = 'The following UGIDs used to limit geometries: {0}'.format(self.value)
        return(msg)
    
    
class VectorWrap(BooleanParameter,AttributedOcgParameter):
    _name = 'vector_wrap'
    _nullable = True
    _default = True

    def message(self):
        if self.value:
            msg = 'Geographic coordinates wrapped from -180 to 180 degrees longitude.'
        else:
            msg = 'Geographic coordinates match the target dataset coordinate wrapping and may be in the range 0 to 360.'
        return(msg)
    
    
class Unwrap(BooleanParameter,AttributedOcgParameter):
    _name = 'unwrap'
    _nullable = True
    _default = False
    
    
class AllowEmpty(BooleanParameter,AttributedOcgParameter):
    _name = 'allow_empty'
    _nullable = True
    _default = False
    
    def message(self):
        if self.value:
            msg = 'Empty returns are allowed. Selection geometries not overlapping with dataset geometries are excluded from a return. Empty output data may results for absolutely no overlap.'
        else:
            msg = 'Emptry returns NOT allowed. If a selection geometry has no intersecting geometries from the target dataset, an exception is raised.'
        return(msg)
    
    
class PrimeMeridian(AttributedOcgParameter):
    _name = 'pm'
    _nullable = True
    _default = 0.0
    _dtype = float
    
    def _format_string_(self,value):
        return(float(value))


## determine the iterator mode for the converters
def identify_iterator_mode(ops):
    '''raw,agg,calc,multi'''
    mode = 'raw'
    if ops.calc is not None:
        mode = 'calc'
        for f in ops.calc:
            if issubclass(f['ref'],OcgCvArgFunction):
                mode = 'multi'
                break
    ops.mode = mode
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()