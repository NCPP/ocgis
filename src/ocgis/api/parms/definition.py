from ocgis.api.parms import base
from ocgis.exc import DefinitionValidationError
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
from shapely.geometry.polygon import Polygon
from collections import OrderedDict
import ocgis
from os.path import exists
from shapely.geometry.multipolygon import MultiPolygon
from types import NoneType
from shapely.geometry.point import Point
from ocgis import constants
from ocgis.util.shp_cabinet import ShpCabinetIterator
from ocgis.calc.library.register import FunctionRegistry
from ocgis.interface.base.crs import CoordinateReferenceSystem, CFWGS84
from ocgis.util.logging_ocgis import ocgis_lh
import logging


class Abstraction(base.StringOptionParameter):
    name = 'abstraction'
    default = None
    valid = ('point','polygon')
    nullable = True
    
    def _get_meta_(self):
        if self.value is None:
            msg = 'Highest order geometry available used for spatial output.'
        else:
            msg = 'Spatial dimension abstracted to {0}.'.format(self.value)
        return(msg)


class AllowEmpty(base.BooleanParameter):
    name = 'allow_empty'
    default = False
    meta_true = 'Empty returns are allowed. Selection geometries not overlapping with dataset geometries are excluded from a return. Empty output data may results for absolutely no overlap.'
    meta_false = 'Empty returns NOT allowed. If a selection geometry has no intersecting geometries from the target dataset, an exception is raised.'

    
class Aggregate(base.BooleanParameter):
    name = 'aggregate'
    default = False
    meta_true = ('Selected geometries are aggregated (unioned), and associated '
                 'data values are area-weighted based on final area following the '
                 'spatial operation. Weights are normalized using the maximum area '
                 'of the geometry set.')
    meta_false = 'Selected geometries are not aggregated (unioned).'
    
    
class AggregateSelection(base.BooleanParameter):
    name = 'agg_selection'
    default = False
    meta_true = 'Selection geometries were aggregated (unioned).'
    meta_false = 'Selection geometries left as is.'


class Backend(base.StringOptionParameter):
    name = 'backend'
    default = 'ocg'
    valid = ('ocg',)
    
    def _get_meta_(self):
        if self.value == 'ocg':
            ret = 'OpenClimateGIS backend used for processing.'
        else:
            raise(NotImplementedError)
        return(ret)
    
    
class Calc(base.IterableParameter,base.OcgParameter):
    name = 'calc'
    default = None
    nullable = True
    input_types = [list,tuple]
    return_type = list
    element_type = dict
    unique = False
    
    def __repr__(self):
        msg = '{0}={1}'.format(self.name,self.value)
        return(msg)
    
    def get_url_string(self):
        raise(NotImplementedError)
#        if self.value is None:
#            ret = 'none'
#        else:
#            elements = []
#            for element in self.value:
#                strings = []
#                template = '{0}~{1}'
#                if element['ref'] != library.SampleSize:
#                    strings.append(template.format(element['func'],element['name']))
#                    for k,v in element['kwds'].iteritems():
#                        strings.append(template.format(k,v))
#                if len(strings) > 0:
#                    elements.append('!'.join(strings))
#            ret = '|'.join(elements)
#        return(ret)
    
    def _get_meta_(self):
        if self.value is None:
            ret = 'No computations applied.'
        else:
            ret = ['The following computations were applied:']
            for ii in self.value:
                ret.append('{0}: {1}'.format(ii['name'],ii['ref'].description))
        return(ret)
    
    def _parse_(self,value):
        fr = FunctionRegistry()
        value['ref'] = fr[value['func']]
        if 'kwds' not in value:
            value['kwds'] = OrderedDict()
        else:
            value['kwds'] = OrderedDict(value['kwds'])
            for k,v in value['kwds'].iteritems():
                try:
                    value['kwds'][k] = v.lower()
                except AttributeError:
                    pass
        return(value)
        
    def _parse_string_(self,value):
        key,uname = value.split('~',1)
        try:
            uname,kwds_raw = uname.split('!',1)
            kwds_raw = kwds_raw.split('!')
            kwds = OrderedDict()
            for kwd in kwds_raw:
                kwd_name,kwd_value = kwd.split('~')
                try:
                    kwds.update({kwd_name:float(kwd_value)})
                except ValueError:
                    kwds.update({kwd_name:str(kwd_value)})
        except ValueError:
            kwds = OrderedDict()
        dct = {'func':key,'name':uname,'kwds':kwds}
        return(dct)
    
    def _validate_(self,value):
        ## collect names
        names = [ii['name'] for ii in value]
        if len(names) != len(set(names)):
            raise(DefinitionValidationError(self,'User-provided calculation names must be unique.'))

    
class CalcGrouping(base.IterableParameter,base.OcgParameter):
    name = 'calc_grouping'
    nullable = True
    input_types = [list,tuple]
    return_type = tuple
    default = None
    element_type = [str,list]
    unique = True
    
    def _get_meta_(self):
        if self.value is None:
            msg = 'No temporal aggregation applied.'
        else:
            msg = 'Temporal aggregation determined by the following group(s): {0}'.format(self.value)
        return(msg)
    
    def _validate_(self,value):
        try:
            for val in value:
                if val not in ['day','month','year']:
                    raise(DefinitionValidationError(self,'"{0}" is not a valid temporal group or is currently not supported. Supported groupings are combinations of day, month, and year.'.format(val)))
        ## the grouping may not be a date part but a seasonal aggregation
        except DefinitionValidationError:
            months = range(1,13)
            for element in value:
                for month in element:
                    if month not in months:
                        raise(DefinitionValidationError(self,'Month integer value is not recognized: {0}'.format(month)))
            
class CalcRaw(base.BooleanParameter):
    name = 'calc_raw'
    default = False
    meta_true = 'Raw values will be used for calculations. These are the original data values linked to a selection geometry.'
    meta_false = 'Aggregated values will be used during the calculation.'


class CalcSampleSize(base.BooleanParameter):
    name = 'calc_sample_size'
    default = False
    meta_true = 'Statistical sample size for calculations added to output files.'
    meta_false = 'Statistical sample size not calculated.'


class Dataset(base.OcgParameter):
    name = 'dataset'
    nullable = False
    default = None
    input_types = [RequestDataset,list,tuple,RequestDatasetCollection,dict]
    return_type = RequestDatasetCollection
    
    def __init__(self,arg):
        if arg is not None:
            if isinstance(arg,RequestDatasetCollection):
                init_value = arg
            else:
                if isinstance(arg,RequestDataset):
                    itr = [arg]
                elif isinstance(arg,dict):
                    itr = [arg]
                else:
                    itr = arg
                rdc = RequestDatasetCollection()
                for rd in itr:
                    rdc.update(rd)
                init_value = rdc
            ## dereference any prior dataset connections
            for rd in init_value:
                rd._ds = None
        else:
            init_value = arg
        super(Dataset,self).__init__(init_value)
        
    def parse_string(self,value):
        lowered = value.strip()
        if lowered == 'none':
            ret = None
        else:
            ret = self._parse_string_(lowered)
        return(ret)
    
    def get_meta(self):
        return(self.value._get_meta_rows_())
    
    def _get_meta_(self): pass
    
    def _parse_string_(self,lowered):
        raise(NotImplementedError)
    
    
class DirOutput(base.StringParameter):
    _lower_string = False
    name = 'dir_output'
    nullable = False
    default = ocgis.env.DIR_OUTPUT
    return_type = str
    input_types = []
    
    def _get_meta_(self):
        ret = 'At execution time, data was originally written to this processor-local location: {0}'.format(self.value)
        return(ret)
    
    def _validate_(self,value):
        if not exists(value):
            raise(DefinitionValidationError(self,'Output directory does not exist: {0}'.format(value)))


class FileOnly(base.BooleanParameter):
    meta_true = 'File written with empty data.'
    meta_false = 'Actual data written to file.'
    default = False
    name = 'file_only'


class FormatTime(base.BooleanParameter):
    name = 'format_time'
    default = True
    meta_true = 'Time values converted to datetime stamps.'
    meta_false = 'Time values left in original form.'


class Geom(base.OcgParameter):
    name = 'geom'
    nullable = True
    default = None
    input_types = [list,tuple,Polygon,MultiPolygon,ShpCabinetIterator]
    return_type = [list,ShpCabinetIterator]
    _shp_key = None
    _bounds = None
    
    def __init__(self,*args,**kwds):
        self.select_ugid = kwds.pop('select_ugid',None)
        args = [self] + list(args)
        base.OcgParameter.__init__(*args,**kwds)
    
    def __str__(self):
        if self.value is None:
            value = None
        elif self._shp_key is not None:
            value = self._shp_key
        elif self._bounds is not None:
            value = '|'.join(map(str,self._bounds))
        else:
            value = '<{0} geometry(s)>'.format(len(self.value))
        ret = '{0}={1}'.format(self.name,value)
        return(ret)
    
    def parse(self,value):
        if type(value) in [Polygon,MultiPolygon,Point]:
            ret = [{'geom':value,'properties':{'ugid':1},'crs':CFWGS84()}]
        elif type(value) in [list,tuple]:
            if all([isinstance(element,dict) for element in value]):
                for ii,element in enumerate(value,start=1):
                    if 'geom' not in element:
                        ocgis_lh(exc=DefinitionValidationError(self,'Geometry dictionaries must have a "geom" key.'))
                    if 'properties' not in element:
                        element['properties'] = {'UGID':ii}
                    if 'crs' not in element:
                        element['crs'] = CFWGS84()
                        ocgis_lh(msg='No CRS in geometry dictionary - assuming WGS84.',level=logging.WARN,check_duplicate=True)
                ret = value
            else:
                if len(value) == 2:
                    geom = Point(value[0],value[1])
                elif len(value) == 4:
                    minx,miny,maxx,maxy = value
                    geom = Polygon(((minx,miny),
                                    (minx,maxy),
                                    (maxx,maxy),
                                    (maxx,miny)))
                if not geom.is_valid:
                    raise(DefinitionValidationError(self,'Parsed geometry is not valid.'))
                ret = [{'geom':geom,'properties':{'ugid':1},'crs':CFWGS84()}]
                self._bounds = geom.bounds
        elif isinstance(value,ShpCabinetIterator):
            self._shp_key = value.key
            ret = value
        else:
            ret = value
        return(ret)
    
    def parse_string(self,value):
        elements = value.split('|')
        try:
            elements = [float(e) for e in elements]
            ## switch geometry creation based on length. length of 2 is a point
            ## otherwise a bounding box
            if len(elements) == 2:
                geom = Point(elements[0],elements[1])
            else:
                minx,miny,maxx,maxy = elements
                geom = Polygon(((minx,miny),
                                (minx,maxy),
                                (maxx,maxy),
                                (maxx,miny)))
            if not geom.is_valid:
                raise(DefinitionValidationError(self,'Parsed geometry is not valid.'))
            ret = [{'geom':geom,'properties':{'ugid':1}}]
            self._bounds = elements
        except ValueError:
            self._shp_key = value
            ## get the select_ugid test value.
            try:
                test_value = self.select_ugid.value
            except AttributeError:
                test_value = self.select_ugid
            if test_value is None:
                select_ugid = None
            else:
                select_ugid = test_value
            ret = ShpCabinetIterator(value,select_ugid=select_ugid)
        return(ret)
    
    def _get_meta_(self):
        if self.value is None:
            ret = 'No user-supplied geometry. All data returned.'
        elif self._shp_key is not None:
            ret = 'The selection geometry "{0}" was used for subsetting.'.format(self._shp_key)
        elif self._bounds is not None:
            ret = 'The bounding box coordinates used for subset are: {0}.'.format(self._bounds)
        else:
            ret = '{0} custom user geometries provided.'.format(len(self.value))
        return(ret)
    
    
class Headers(base.IterableParameter,base.OcgParameter):
    name = 'headers'
    default = None
    return_type = tuple
    valid = set(constants.raw_headers+constants.calc_headers+constants.multi_headers)
    input_types = [list,tuple]
    nullable = True
    element_type = str
    unique = True

    def __repr__(self):
        try:
            msg = '{0}={1}'.format(self.name,self.split_string.join(self.value))
        ## likely a NoneType
        except TypeError:
            if self.value is None:
                msg = '{0}=none'.format(self.name)
            else:
                raise
        return(msg)
    
    def parse_all(self,value):
        for header in constants.required_headers:
            if header in value:
                value.remove(header)
        return(constants.required_headers+value)
    
    def validate_all(self,values):
        if len(values) == 0:
            msg = 'At least one header value must be passed.'
            raise(DefinitionValidationError(self,msg))
        if not self.valid.issuperset(values):
            msg = 'Valid headers are {0}.'.format(list(self.valid))
            raise(DefinitionValidationError(self,msg))

    def _get_meta_(self):
        return('The following headers were used for file creation: {0}'.format(self.value))


class OutputCRS(base.OcgParameter):
    input_types = [CoordinateReferenceSystem]
    name = 'output_crs'
    nullable = True
    return_type = [CoordinateReferenceSystem]
    default = None
    
    def _get_meta_(self):
        if self.value is None:
            ret = "No CRS associated with dataset. WGS84 Lat/Lon Geographic (EPSG:4326) assumed."
        else:
            ret = 'The PROJ.4 definition of the coordinate reference system is: "{0}"'.format(self.value.sr.ExportToProj4())
        return(ret)


class OutputFormat(base.StringOptionParameter):
    name = 'output_format'
    default = 'numpy'
    valid = constants.output_formats
    
    def _get_meta_(self):
        ret = 'The output format is "{0}".'.format(self.value)
        return(ret)
    
    
class Prefix(base.StringParameter):
    name = 'prefix'
    nullable = False
    default = 'ocgis_output'
    input_types = [str]
    return_type = str
    _lower_string = False
    
    def _get_meta_(self):
        msg = 'Data output given the following prefix: {0}.'.format(self.value)
        return(msg)
    

class SearchRadiusMultiplier(base.OcgParameter):
    input_types = [float]
    name = 'search_radius_mult'
    nullable = False
    return_type = [float]
    
    def _get_meta_(self):
        msg = 'If point geometries were used for selection, a modifier of {0} times the data resolution was used to spatially select data.'.format(self.value)
        return(msg)
    
    def _validate_(self,value):
        if value <= 0:
            raise(DefinitionValidationError(self,msg='must be >= 0'))
    

class SelectUgid(base.IterableParameter,base.OcgParameter):
    name = 'select_ugid'
    return_type = tuple
    nullable = True
    default = None
    input_types = [list,tuple]
    element_type = int
    unique = True
    
    def _get_meta_(self):
        if self.value is None:
            ret = 'No geometry selection by unique identifier.'
        else:
            ret = 'The following UGID values were used to select from the input geometries: {0}.'.format(self.value)
        return(ret)
    
    
class Slice(base.IterableParameter,base.OcgParameter):
    name = 'slice'
    return_type = tuple
    nullable = True
    default = None
    input_types = [list,tuple]
    element_type = [NoneType,int,tuple,list,slice]
    unique = False
    
    def validate_all(self,values):
        if len(values) != 5:
            raise(DefinitionValidationError(self,'Slices must have 5 values.'))
    
    def _parse_(self,value):
        if value is None:
            ret = slice(None)
        elif type(value) == int:
            ret = slice(value,value+1)
        elif type(value) in [list,tuple]:
            ret = slice(*value)
        else:
            raise(DefinitionValidationError(self,'"{0}" cannot be converted to a slice object'.format(value)))
        return(ret)
    
    def _get_meta_(self):
        if self.value is None:
            ret = 'No slice passed.'
        else:
            ret = 'A slice was used.'
        return(ret)


class Snippet(base.BooleanParameter):
    name = 'snippet'
    default = False
    meta_true = 'First temporal slice or temporal group returned.'
    meta_false = 'All time points returned.'
    
    
class SpatialOperation(base.StringOptionParameter):
    name = 'spatial_operation'
    default = 'intersects'
    valid = ('clip','intersects')
    
    def _get_meta_(self):
        if self.value == 'intersects':
            ret = 'Geometries touching AND overlapping returned.'
        else:
            ret = 'A full geometric intersection occurred. Where geometries overlapped, a new geometry was created.'
        return(ret)


class VectorWrap(base.BooleanParameter):
    name = 'vector_wrap'
    default = True
    meta_true = 'Geographic coordinates wrapped from -180 to 180 degrees longitude.'
    meta_false = 'Geographic coordinates match the target dataset coordinate wrapping and may be in the range 0 to 360.'
