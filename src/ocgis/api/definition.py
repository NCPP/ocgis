from datetime import datetime
from copy import deepcopy
from types import NoneType
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from ocgis.calc.base import OcgFunctionTree
from ocgis.calc import library
from ocgis.exc import DefinitionValidationError, CannotEncodeUrl
from collections import OrderedDict


class OcgParameter(object):
    
    def __init__(self,name,dtype,nullable=False,default=None,length=None,
                 alias=None,init_value=None,scalar=True):
        self.name = name
        self.nullable = nullable
        self.default = default
        self.dtype = dtype
        self.length = length
        self.alias = alias
        self.scalar = scalar
        
        self._value = None
        if init_value is not None:
            self.value = init_value
        else:
            self.value = None
            
    def __str__(self):
        msg = '{0}={1}'
        ret = self._to_str_()
        ret = ret.lower()
        msg = msg.format(self.name,ret)
        return(msg)
    
    def _to_str_(self):
        return(str(self.value))
       
    @property
    def value(self):
        if self._value is None or self._value == [None]:
            ret = None
        elif self.scalar:
            ret = self._value[0]
        else:
            ret = self._value
        return(ret)
    @value.setter
    def value(self,value):
        value = deepcopy(value)
        self._value = self.format(value)
        
    def format(self,value):
        if value is None or isinstance(value,basestring) or isinstance(value,dict):
            it = [value]
        else:
            it = self._get_iter_(value)
        
        try:
            ret = []
            for val in it:
                if val is None:
                    if self.default is None and self.nullable is False:
                        raise(ValueError('"{0}" is not nullable.'.format(self.name)))
                    else:
                        app = self.default
                else:
                    app = self._format_element_(val)
                ret.append(app)
        except:
            ret = self.format_string(value)
            self.value = ret
        if not all([ii == None for ii in ret]):
            ret = self.format_all(ret)
            self.validate_all(ret)
        return(ret)
    
    def format_all(self,values):
        return(values)
    
    def _format_all_(self,values):
        return(values)
    
    def _get_iter_(self,value):
        try:
            it = iter(value)
        except TypeError:
            it = [value]
        return(it)
    
    def _format_element_(self,element):
        ret = self._format_(element)
        try:
            self._assert_(type(ret) == self.dtype,'Data types do not match.')
        except:
            try:
                self._assert_(type(ret) in self.dtype,'Data types do not match.')
            except:
                raise
        if self.length is not None:
            self._assert_(len(ret) == self.length,'Lengths do not match.')
        self.validate(ret)
        return(ret)
    
    def format_string(self,value):

        def _format_(value):
            value = value.lower()
            if value == 'none':
                ret = None
            else:
                ret = value
            return(ret)
        
        try:
            it = value.split('|')
        except AttributeError:
            if isinstance(value[0],basestring):
                it = value
            else:
                raise
        lowered = map(_format_,it)
        
        ret = []
        for val in lowered:
            if val is None:
                app = val
            else:
                app = self._format_string_element_(val)
                app = self._format_element_(app)
                self.validate(app)
            ret.append(app)
        return(ret)
    
    def parse_query(self,query):
        try:
            value = query[self.name]
        except KeyError:
            value = query.get(self.alias)
        if value is None:
            to_set = None
        else:
            value = value[0]
            to_set = value
        self.value = to_set
        
    def validate(self,value):
        pass
    
    def validate_all(self,values):
        pass
    
    def message(self):
        raise(NotImplementedError)
    
    def _assert_(self,test,msg=None):
        try:
            assert(test)
        except AssertionError:
            raise(DefinitionValidationError(self,msg))
    
    def _format_(self,value):
        return(value)
    
    def _format_string_element_(self,value):
        return(value)
    
    
class BooleanParameter(OcgParameter):
    _dtype = bool
    
    def _format_string_element_(self,value):
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
    _scalar = True
    
    def __init__(self,init_value=None):
        super(AttributedOcgParameter,self).__init__(
         self._name,self._dtype,nullable=self._nullable,default=self._default,
         length=self._length,alias=self._alias,init_value=init_value,scalar=self._scalar)
    
    
class RequestUrl(AttributedOcgParameter):
    _name = 'request_url'
    _nullable = True
    _dtype = str
    
    def message(self):
        msg = 'Requested URL:\n{0}'.format(self.value)
        return(msg)
    
    
class Snippet(BooleanParameter,AttributedOcgParameter):
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
    _name = 'calc_grouping'
    _nullable = True
    _dtype = str
    _scalar = False
    
    def validate(self,value):
        self._assert_(value in ['day','month','year','hour','minute','second'],'"{0}" is not a valid group.'.format(value))
    
    def _format_string_element_(self,value):
        grouping = value.split('|')
        return(list(grouping))
    
    def message(self):
        msg = ('Temporal aggregation determined by the following group(s): {0}')
        msg = msg.format(self.value)
        return(msg)
    
    
class OutputFormat(AttributedOcgParameter):
    _possible = ['numpy','shpidx','shp','csv','nc','keyed','meta']
    _name = 'output_format'
    _nullable = True
    _default = 'numpy'
    _dtype = str
    
    def message(self):
        mmap = {'numpy':'an OpenClimateGIS data format storing variables as NumPy arrays.',
                'shpidx':'an ESRI Shapefile mapping unique geometry identifiers (GID) to geometries constructed from the dataset.',
                'shp':'an ESRI Shapefile containing all output data.',
                'csv':'a Comma Separated Value files containing all output data.',
                'keyed':'a group of keyed files linking to the "value" file. Files are named after the unique key for which they contain data. A geometry index is also included.',
                'meta':'a description of the operations performed or to be performed given provided parameters. No data is touched during the operation. Hence, operations may fail if dataset and/or parameters are somehow noncompliant.',
                'nc':'a NetCDF4 file.'
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
    
    
class Geom(OcgParameter):
    
    def __init__(self,init_value=None):
        self._default = [{'geom':None,'ugid':1}]
        self._shp_key = None
        self._bounds = None
        super(self.__class__,self).__init__('geom',(dict,list),init_value=init_value,
                                            nullable=True,default=self._default,scalar=True)
        
    def _to_str_(self):
        if self.value == self._default:
            ret = 'none'
        elif self._shp_key is not None:
            ret = self._shp_key
        elif self._bounds is not None:
            ret = '|'.join(self._bounds)
        else:
            raise(CannotEncodeUrl('Too many geometries to encode.'))
        return(ret)
    
    def _format_(self,value):
        ## first try to format using the string. geometry names can be passed
        ## this way.
        try:
            ret = self._format_string_element_(value)
            self._shp_key = value
        except AttributeError:
            try:
                concat = '|'.join(map(str,value))
                ret = self._format_string_element_(concat)
            except:
                ret = value
        return(ret)
    
    def validate(self,value):
        if isinstance(value,dict):
            value = [value]
        for v in value: 
            self._assert_(type(v) == dict,
             'list elements must be dictionaries with keys "ugid" and "geom"')
            self._assert_('ugid' in v,'a geom must have a "ugid" key')
            self._assert_('geom' in v,'a geom dict must have a "geom" key')
            self._assert_(type(v['geom']) in [NoneType,Polygon,MultiPolygon])
    
    def message(self):
        for ii in self.value:
            if ii['geom'] is None:
                msg = 'No user-supplied geometry. All data returned.'
                return(msg)
            else:
                msg = '{0} user geometries provided.'.format(len(self.value))
                return(msg)
            
    def _format_string_element_(self,value):
        elements = value.split('|')
        try:
            elements = [float(e) for e in elements]
            minx,miny,maxx,maxy = elements
            geom = Polygon(((minx,miny),
                            (minx,maxy),
                            (maxx,maxy),
                            (maxx,miny)))
            self._assert_(geom.is_valid)
            ret = [{'ugid':1,'geom':geom}]
            self._bounds = elements
        except ValueError:
            from ocgis.util.shp_cabinet import ShpCabinet
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
    
    def _get_iter_(self,value):
        ret = value
        if type(value) in (list,tuple) and isinstance(value[0],dict):
            ret = [value]
        else:
            try:
                if all([type(ii) in (float,int) for ii in value]):
                    ret = [value]
            except:
                pass
        return(ret)


class Calc(AttributedOcgParameter):
    _name = 'calc'
    _nullable = True
    _dtype = dict
    _default = None
    _scalar = False
    
    def _format_(self,value):
        potentials = OcgFunctionTree.get_potentials()
        for p in potentials:
            if p[0] == value['func']:
                value['ref'] = getattr(library,p[1])
                break
        if 'kwds' not in value:
            value['kwds'] = {}
        else:
            for k,v in value['kwds'].iteritems():
                try:
                    value['kwds'][k] = v.lower()
                except AttributeError:
                    pass
        return(value)
    
    def format_all(self,values):
        values.append(self._format_({'func':'n','name':'n'}))
        return(values)
    
    def _get_iter_(self,values):
        if values == None:
            ret = [values]
        elif isinstance(values,dict):
            ret = [values]
        else:
            ret = values
        return(ret)
    
    def _format_string_element_(self,value):
        key,uname = value.split('~',1)
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
        return(dct)
    
    def validate(self,value):
        self._assert_('func' in value,('The function name is '
                                       'required using the key "func"'))
        self._assert_('name' in value,'A custom function "name" is required.')
            
    def validate_all(self,values):
        names = [ii['name'] for ii in values]
        self._assert_(len(set(names)) == len(names),'Function names must be unique.')
            
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


class RequestDataset(object):
    '''A :class:`RequestDataset` contains all the information necessary to find
    and subset a variable (by time and/or level) contained in a local or 
    OpenDAP-hosted CF dataset.
    
    >>> from ocgis import RequestDataset
    >>> uri = 'http://some.opendap.dataset
    >>> variable = 'tasmax'
    >>> rd = RequestDataset(uri,variable)
    
    :param uri: The absolute path (URLs included) to the dataset's location.
    :type uri: str
    :param variable: The target variable.
    :type variable: str
    :param alias: An alternative name to identify the returned variable's data. If `None`, this defaults to `variable`. If variables having the same name occur in a request, this value will be required.
    :type alias: str
    :param time_range: Upper and lower bounds for time dimension subsetting. If `None`, return all time points.
    :type time_range: [:class:`datetime.datetime`, :class:`datetime.datetime`]
    :param level_range: Upper and lower bounds for level dimension subsetting. If `None`, return all levels.
    :type level_range: [int, int]
    :param s_proj: A `PROJ4 string`_ describing the dataset's spatial reference.
    :type s_proj: str
    :param t_units: Overload the autodiscover `time units`_.
    :type t_units: str
    :param t_calendar: Overload the autodiscover `time calendar`_.
    :type t_calendar: str
    
    .. _PROJ4 string: http://trac.osgeo.org/proj/wiki/FAQ
    .. _time units: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    .. _time calendar: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    '''
    
    def __init__(self,uri,variable,alias=None,time_range=None,level_range=None,
                 s_proj=None,t_units=None,t_calendar=None):
        self.uri = uri
        self.variable = variable
        self.alias = self._str_format_(alias) or variable
        self.time_range = deepcopy(time_range)
        self.level_range = deepcopy(level_range)
        self.s_proj = self._str_format_(s_proj)
        self.t_units = self._str_format_(t_units)
        self.t_calendar = self._str_format_(t_calendar)
        
        self.ocg_dataset = None
        self._use_for_id = []
        self._format_()
        
    @property
    def interface(self):
        attrs = ['s_proj','t_units','t_calendar']
        ret = {attr:getattr(self,attr) for attr in attrs}
        return(ret)
        
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return(self.__dict__ == other.__dict__)
        else:
            return(False)
        
    def __repr__(self):
        msg = '<{0} ({1})>'.format(self.__class__.__name__,self.alias)
        return(msg)
    
    def __getitem__(self,item):
        ret = getattr(self,item)
        return(ret)
    
    def _str_format_(self,value):
        ret = value
        if isinstance(value,basestring):
            value = value.lower()
            if value == 'none':
                ret = None
        else:
            ret = value
        return(ret)
    
    def _format_(self):
        if self.time_range is not None:
            self._format_time_range_()
        if self.level_range is not None:
            self._format_level_range_()
    
    def _format_time_range_(self):
        try:
            ret = [datetime.strptime(v,'%Y-%m-%d') for v in self.time_range.split('|')]
        except AttributeError:
            ret = self.time_range
        ref = ret[1]
        ret[1] = datetime(ref.year,ref.month,ref.day,23,59,59)
        if ret[0] > ret[1]:
            raise(DefinitionValidationError('dataset','Time ordination incorrect.'))
        self.time_range = ret
        
    def _format_level_range_(self):
        try:
            ret = [int(v) for v in self.level_range.split('|')]
        except AttributeError:
            ret = self.level_range
        if ret[0] > ret[1]:
            raise(DefinitionValidationError('dataset','Level ordination incorrect.'))
        self.level_range = ret
    
    
class RequestDatasetCollection(object):
    '''Contains business logic ensuring multi-:class:`ocgis.RequestDataset` objects are
    compatible.
    
    >>> from ocgis import RequestDatasetCollection, RequestDataset
    >>> uris = ['http://some.opendap.dataset1', 'http://some.opendap.dataset2']
    >>> variables = ['tasmax', 'tasmin']
    >>> request_datasets = [RequestDatset(uri,variable) for uri,variable in zip(uris,variables)]
    >>> rdc = RequestDatasetCollection(request_datasets)
    
    ## Update object in place.
    >>> rdc = RequestDatasetCollection()
    >>> for rd in request_datasets:
    ...     rdc.update(rd)
    
    :param request_datasets: A sequence of :class:`ocgis.RequestDataset` objects.
    :type request_datasets: sequence of :class:`ocgis.RequestDataset` objects
    '''
    
    def __init__(self,request_datasets=[]):
        self._s = OrderedDict()
        for rd in request_datasets:
            self.update(rd)
            
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return(self.__dict__ == other.__dict__)
        else:
            return(False)
        
    def __len__(self):
        return(len(self._s))
        
    def __repr__(self):
        msg = ['{0} RequestDataset(s) in collection:'.format(len(self))]
        for rd in self:
            msg.append('  {0}'.format(rd.__repr__()))
        return('\n'.join(msg))
        
    def __iter__(self):
        for value in self._s.itervalues():
            yield(value)
            
    def __getitem__(self,index):
        try:
            ret = self._s[index]
        except KeyError:
            key = self._s.keys()[index]
            ret = self._s[key]
        return(ret)
    
    def update(self,request_dataset):
        """Add a :class:`ocgis.RequestDataset` to the collection.
        
        :param request_dataset: The :class:`ocgis.RequestDataset` to add.
        :type request_dataset: :class:`ocgis.RequestDataset`
        """
        if request_dataset.alias in self._s:
            raise(KeyError('Alias "{0}" already in collection.'\
                           .format(request_dataset.alias)))
        else:
            self._s.update({request_dataset.alias:request_dataset})
            
    
class Dataset(OcgParameter):
    
    def __init__(self,init_value=None):
        self._coll = RequestDatasetCollection()
        super(self.__class__,self).__init__('dataset',RequestDataset,
                 nullable=False,default=None,length=None,
                 alias=None,init_value=init_value,scalar=False)
    
    def format(self,value):
        value = deepcopy(value)
        if isinstance(value,RequestDataset):
            self._coll.update(value)
            ret = self._coll
        elif isinstance(value,RequestDatasetCollection):
            ret = value
        else:
            ret = super(self.__class__,self).format(value)
        return(ret)    
    
    def __str__(self):
        if len(self.value) == 1:
            end_integer_strings = ['']
        else:
            end_integer_strings = range(1,len(self.value)+1)
        out_str = []
        template = '{0}{1}={2}'
        for ds,es in zip(self.value,end_integer_strings):
            for key in ['uri','variable','alias','t_units','t_calendar','s_proj']:
                app_value = ds[key]
                if app_value is None:
                    app_value = 'none'
                app = template.format(key,es,app_value)
                out_str.append(app)
        out_str = '&'.join(out_str)
        return(out_str)
    
    @classmethod
    def parse_query(cls,query):
        
        def _get_ref_(query,key):
            ret = query.get(key)
            if ret is not None:
                if not isinstance(ret[0],basestring):
                    ret = ret[0]
            return(ret)
        
        refs = {}
        keys = ['uri','variable','alias','t_units','t_calendar','s_proj','time_range','level_range']
        for key in keys:
            refs.update({key:_get_ref_(query,key)})
        ret = []
        for idx in range(len(refs['uri'])):
            app = {}
            for key in keys:
                try:
                    ref = refs[key][idx]
                except TypeError:
                    ref = None
                app.update({key:ref})
            ret.append(app)
        return(cls(ret))
    
    def _format_element_(self,value):
        rd = RequestDataset(**value)
        return(rd)
    
    def format_all(self,values):
        for rd in values:
            self._coll.update(rd)
        return(self._coll)
    
    def message(self):
        lines = []
        for ii in self.value:
            lines.append('* Variable "{0}" requested from dataset with URI "{1}".'.format(ii['variable'],ii['uri']))
        return('\n'.join(lines))
    
    
class Abstraction(AttributedOcgParameter):
    _name = 'abstraction'
    _nullable = True
    _dtype = str
    _default = 'polygon'
    _scalar = True
    
    def validate(self,value):
        valid = ['point','polygon']
        self._assert_(value in valid,"'{0}' not in {1}".format(value,valid))
    
    def message(self):
        msg = 'Spatial dimension abstracted to {0}.'.format(self.value)
        return(msg)
    
    
class SelectUgid(AttributedOcgParameter):
    _name = 'select_ugid'
    _nullable = True
    _default = None
    _dtype = dict
    
    def _format_string_element_(self,value):
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
    
    def _format_string_element_(self,value):
        return(float(value))


## determine the iterator mode for the converters
def identify_iterator_mode(ops):
    '''raw,agg,calc,multi'''
    mode = 'raw'
    if ops.calc is not None:
        mode = 'calc'
    ops.mode = mode
