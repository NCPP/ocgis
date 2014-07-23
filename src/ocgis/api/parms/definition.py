from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry
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
from ocgis.interface.base.dimension.spatial import SpatialDimension
from ocgis.util.helpers import make_poly, iter_array
from ocgis.util.shp_cabinet import ShpCabinetIterator, ShpCabinet
from ocgis.calc.library import register
from ocgis.interface.base.crs import CoordinateReferenceSystem, CFWGS84
from ocgis.util.logging_ocgis import ocgis_lh
import logging
import os
from copy import deepcopy
from types import FunctionType
import itertools
from ocgis.calc.eval_function import EvalFunction, MultivariateEvalFunction
import datetime


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
    
    
class AddAuxiliaryFiles(base.BooleanParameter):
    name = 'add_auxiliary_files'
    default = True
    meta_true = 'Auxiliary metadata files added. A new directory was created.'
    meta_false = 'Auxiliary metadata files not added. Output file created in the output directory.'


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
    
    
class Callback(base.OcgParameter):
    input_types = [FunctionType]
    name = 'callback'
    nullable = True
    default = None
    return_type = [FunctionType]
    
    def _get_meta_(self):
        if self.value is None:
            msg = 'No callback function provided.'
        else:
            msg = 'Callback enabled.'
        return(msg)
    
    
class Calc(base.IterableParameter,base.OcgParameter):
    name = 'calc'
    default = None
    nullable = True
    input_types = [list,tuple]
    return_type = [list]
    element_type = [dict,str]
    unique = False
    _possible = ['es=tas+4',['es=tas+4'],[{'func':'mean','name':'mean'}]]
    
    def __init__(self,*args,**kwargs):
        ## this flag is used by the parser to determine if an eval function has
        ## been passed. very simple test for this...if there is an equals sign
        ## in the string then it is considered an eval function
        self._is_eval_function = False
        base.OcgParameter.__init__(self,*args,**kwargs)
    
    def __str__(self):
        if self.value is None:
            ret = base.OcgParameter.__str__(self)
        else:
            cb = deepcopy(self.value)
            for ii in cb:
                ii.pop('ref')
            ret = '{0}={1}'.format(self.name, cb)
        return ret
    
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
            if self._is_eval_function:
                ret = 'An string function representation was used for calculation: "{0}"'.format(self.value[0])
            else:
                ret = ['The following computations were applied:']
                for ii in self.value:
                    ret.append('{0}: {1}'.format(ii['name'],ii['ref'].description))
        return(ret)
    
    def _parse_(self,value):
        ## if this is not an eval function (a string to interpret as a function)
        ## then construct the function dictionaries. otherwise, pass through
        if '=' in value:
            self._is_eval_function = True
            if EvalFunction.is_multivariate(value):
                eval_klass = MultivariateEvalFunction
            else:
                eval_klass = EvalFunction
            value = {'func':value,'ref':eval_klass}
        ## if it is not an eval function, then do the standard argument parsing
        if not self._is_eval_function:
            fr = register.FunctionRegistry()
            
            ## get the function key string form the calculation definition dictionary
            function_key = value['func']
            ## this is the message for the DefinitionValidationError if this key
            ## may not be found.
            dve_msg = 'The function key "{0}" is not available in the function registry.'.format(function_key)
            
            ## retrieve the calculation class reference from the function registry
            try:
                value['ref'] = fr[function_key]
            ## if the function cannot be found, it may be part of a contributed
            ## library of calculations not registered by default as the external
            ## library is an optional dependency.
            except KeyError:
                ## this will register the icclim indices.
                if function_key.startswith('{0}_'.format(constants.prefix_icclim_function_key)):
                    register.register_icclim(fr)
                else:
                    raise(DefinitionValidationError(self,dve_msg))
            ## make another attempt to register the function
            try:
                value['ref'] = fr[function_key]
            except KeyError:
                raise(DefinitionValidationError(self,dve_msg))
            
            ## parameters will be set to empty if none are present in the calculation
            ## dictionary.
            if 'kwds' not in value:
                value['kwds'] = OrderedDict()
            ## make the keyword parameter definitions lowercase.
            else:
                value['kwds'] = OrderedDict(value['kwds'])
                for k,v in value['kwds'].iteritems():
                    try:
                        value['kwds'][k] = v.lower()
                    except AttributeError:
                        pass
        return(value)
        
    def _parse_string_(self,value):
        try:
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
            ret = {'func':key,'name':uname,'kwds':kwds}
        except ValueError:
            ## likely a string to use for an eval function
            if '=' not in value:
                msg = 'String may not be parsed: "{0}".'.format(value)
                raise(DefinitionValidationError(self,msg))
            else:
                self._is_eval_function = True
                ret = value
            
        return(ret)
    
    def _validate_(self,value):
        if not self._is_eval_function:
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
    _flags = ('unique','year')
    _standard_groups = ('day','month','year')
    
    @classmethod
    def iter_possible(cls):
        standard_seasons = [[3,4,5],[6,7,8],[9,10,11],[12,1,2]]
        for r in [1,2,3]:
            for combo in itertools.combinations(cls._standard_groups,r):
                yield(combo)
        for one in ['all']:
            yield(one)
        flags = list(cls._flags) + [None]
        for flag in flags:
            if flag is not None:
                yld = deepcopy(standard_seasons)
                yld.insert(0,flag)
            else:
                yld = standard_seasons
            yield(yld)
    
    def parse(self,value):
        try:
            ## not interested in looking for unique letters in the "_flags"
            parse_value = list(deepcopy(value))
            ## if we do remove a flag, be sure and append it back
            add_back = None
            for flag in self._flags:
                if flag in parse_value:
                    parse_value.remove(flag)
                    add_back = flag
            ## call superclass method to parse the value for iteration
            ret = base.IterableParameter.parse(self,parse_value,check_basestrings=False)
            ## add the value back if it has been set
            if add_back is not None:
                ret.append(add_back)
        ## value is likely a NoneType
        except TypeError as e:
            if value is None:
                ret = None
            else:
                raise(e)
        return(ret)
    
    def finalize(self):
        if self._value == ('all',):
            self._value = 'all'
    
    def _get_meta_(self):
        if self.value is None:
            msg = 'No temporal aggregation applied.'
        else:
            msg = 'Temporal aggregation determined by the following group(s): {0}'.format(self.value)
        return(msg)
    
    def _validate_(self,value):
        ## the 'all' parameter will be reduced to a string eventually
        if len(value) == 1 and value[0] == 'all':
            pass
        else:
            try:
                for val in value:
                    if val not in self._standard_groups:
                        raise(DefinitionValidationError(self,'"{0}" is not a valid temporal group or is currently not supported. Supported groupings are combinations of day, month, and year.'.format(val)))
            ## the grouping may not be a date part but a seasonal aggregation
            except DefinitionValidationError:
                months = range(1,13)
                for element in value:
                    ## the keyword year and unique are okay for seasonal aggregations
                    if element in self._flags:
                        continue
                    elif isinstance(element,basestring):
                        if element not in self._flags:
                            raise(DefinitionValidationError(self,'Seasonal flag not recognized: "{0}".'.format(element)))
                    else:
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


class ConformUnitsTo(base.OcgParameter):
    name = 'conform_units_to'
    nullable = True
    default = None
    return_type = str
    input_types = []

    def __init__(self, init_value=None):
        ## cfunits is an optional installation. account for this on the import types.
        try:
            from cfunits import Units
            self.input_types.append(Units)
            self.return_type = [self.return_type] + [Units]
        except ImportError:
            pass
        super(ConformUnitsTo, self).__init__(init_value=init_value)

    def _get_meta_(self):
        if self.value is None:
            ret = 'Units were not conformed.'
        else:
            ret = 'Units of all requested datasets were conformed to: "{0}".'.format(self.value)
        return(ret)


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
    input_types = [list, tuple, ShpCabinetIterator, Polygon, MultiPolygon, Point, MultiPoint, SpatialDimension]
    return_type = [ShpCabinetIterator, tuple]
    _shp_key = None
    _bounds = None
    _ugid_key = 'UGID'
    
    def __init__(self,*args,**kwds):
        self.select_ugid = kwds.pop('select_ugid',None)
        ## just store the value if it is a parameter object
        if isinstance(self.select_ugid,SelectUgid):
            self.select_ugid = self.select_ugid._value
        
        args = [self] + list(args)
        base.OcgParameter.__init__(*args,**kwds)
    
    def __str__(self):
        if self.value is None:
            value = None
        elif self._shp_key is not None:
            value = '"{0}"'.format(self._shp_key)
        elif self._bounds is not None:
            value = '|'.join(map(str,self._bounds))
        else:
            value = '<{0} geometry(s)>'.format(len(self.value))
        ret = '{0}={1}'.format(self.name,value)
        return(ret)
    
    def _get_value_(self):
        if isinstance(self._value,ShpCabinetIterator):
            self._value.select_ugid = self.select_ugid
        return(base.OcgParameter._get_value_(self))
    value = property(_get_value_,base.OcgParameter._set_value_)
    
    def parse(self, value):
        if type(value) in [list, tuple]:
            if all([isinstance(element,dict) for element in value]):
                for ii,element in enumerate(value,start=1):
                    if 'geom' not in element:
                        ocgis_lh(exc=DefinitionValidationError(self,'Geometry dictionaries must have a "geom" key.'))
                    if 'properties' not in element:
                        element['properties'] = {self._ugid_key:ii}
                    crs = element.get('crs', CFWGS84())
                    if 'crs' not in element:
                        ocgis_lh(msg='No CRS in geometry dictionary - assuming WGS84.',level=logging.WARN,check_duplicate=True)
                ret = SpatialDimension.from_records(value, crs=crs)
            else:
                if len(value) == 2:
                    geom = Point(value[0], value[1])
                elif len(value) == 4:
                    minx, miny, maxx, maxy = value
                    geom = Polygon(((minx, miny), (minx, maxy), (maxx, maxy), (maxx,  miny)))
                if not geom.is_valid:
                    raise(DefinitionValidationError(self, 'Parsed geometry is not valid.'))
                ret = [{'geom': geom, 'properties': {self._ugid_key: 1}}]
                ret = SpatialDimension.from_records(ret, crs=CFWGS84())
                self._bounds = geom.bounds
        elif isinstance(value, ShpCabinetIterator):
            self._shp_key = value.key or value.path
            # always want to yield SpatialDimension objects
            value.as_spatial_dimension = True
            ret = value
        elif isinstance(value, BaseGeometry):
            ret = [{'geom': value, 'properties': {self._ugid_key: 1}}]
            ret = SpatialDimension.from_records(ret, crs=CFWGS84())
        elif value is None:
            ret = value
        elif isinstance(value, SpatialDimension):
            ret = value
        else:
            raise NotImplementedError(type(value))

        # convert to a tuple if this is a SpatialDimension object
        if isinstance(ret, SpatialDimension):
            ret = tuple(self._iter_spatial_dimension_tuple(ret))

        return ret
    
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
        ## try the value as a key or path
        except ValueError:
            ## if the path exists, then assume it is a path to a shapefile,
            ## otherwise assume it is a key
            kwds = {}
            if os.path.exists(value):
                kwds['path'] = value
            else:
                kwds['key'] = value
            ## this is saved for later use by the openclimategis metadata output
            ## as the input value is inherently transformed
            self._shp_key = value
            ## get the select_ugid test value
            try:
                test_value = self.select_ugid.value
            ## it may not have been passed as a parameter object
            except AttributeError:
                test_value = self.select_ugid
            if test_value is None:
                select_ugid = None
            else:
                select_ugid = test_value
            kwds['select_ugid'] = select_ugid
            ret = ShpCabinetIterator(**kwds)
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

    @staticmethod
    def _iter_spatial_dimension_tuple(spatial_dimension):
        """
        :param spatial_dimension:
        :type spatial_dimension: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :rtype: tuple
        """

        row_range = range(spatial_dimension.shape[0])
        col_range = range(spatial_dimension.shape[1])
        for row, col in itertools.product(row_range, col_range):
            yield spatial_dimension[row, col]

    
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


class InterpolateSpatialBounds(base.BooleanParameter):
    name = 'interpolate_spatial_bounds'
    default = False
    meta_true = 'If no bounds are present on the coordinate variables, an attempt will be made to interpolate boundary polygons.'
    meta_false = 'If no bounds are present on the coordinate variables, no attempt will be made to interpolate boundary polygons.'


class LevelRange(base.IterableParameter,base.OcgParameter):
    name = 'level_range'
    element_type = [int,float]
    nullable = True
    input_types = [list,tuple]
    return_type = tuple
    unique = False
    default = None
    
    def validate_all(self,value):
        if len(value) != 2:
            msg = 'There must be two elements in the sequence.'
            raise(DefinitionValidationError(self,msg))
        if value[0] > value[1]:
            msg = 'The second element must be >= the first element.'
            raise(DefinitionValidationError(self,msg))
    
    def _get_meta_(self):
        if self.value == None:
            msg = 'No level subset.'
        else:
            msg = 'The following level subset was applied to all request datasets: {0}'.format(self.value)
        return(msg)


class Optimizations(base.OcgParameter):
    name = 'optimizations'
    default = None
    input_types = [dict]
    nullable = True
    return_type = [dict]
    #: 'tgds' - dictionary mapping field aliases to TemporalGroupDimension objects
    _allowed_keys = ['tgds','fields']
    _perform_deepcopy = False
    
    def _get_meta_(self):
        if self.value is None:
            ret = 'No optimizations were used.'
        else:
            ret = 'The following optimizations were used: {0}.'.format(self.value.keys())
        return(ret)
    
    def _validate_(self,value):
        if len(value) == 0:
            msg = 'Empty dictionaries are not allowed for optimizations. Use None instead.'
            raise(DefinitionValidationError(self,msg))
        if set(value.keys()).issubset(set(self._allowed_keys)) == False:
            msg = 'Allowed optimization keys are "{0}".'.format(self._allowed_keys)
            raise(DefinitionValidationError(self,msg))


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

    @classmethod
    def iter_possible(cls):
        from ocgis.conv.base import AbstractConverter
        mmap = AbstractConverter.get_converter_map()
        for key in mmap.keys():
            yield key

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
    default = 2.0
    
    def _get_meta_(self):
        msg = 'If point geometries were used for selection, a modifier of {0} times the data resolution was used to spatially select data.'.format(self.value)
        return(msg)
    
    def _validate_(self,value):
        if value <= 0:
            raise(DefinitionValidationError(self,msg='must be >= 0'))


class SelectNearest(base.BooleanParameter):
    name = 'select_nearest'
    default = False
    meta_true = 'The nearest geometry to the centroid of the selection geometry was returned.'
    meta_false = 'All geometries returned regardless of distance.'
    

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
    valid = ('clip', 'intersects')

    @classmethod
    def iter_possible(cls):
        for v in cls.valid:
            yield(v)
    
    def _get_meta_(self):
        if self.value == 'intersects':
            ret = 'Geometries touching AND overlapping returned.'
        else:
            ret = 'A full geometric intersection occurred. Where geometries overlapped, a new geometry was created.'
        return(ret)


class TimeRange(base.IterableParameter,base.OcgParameter):
    name = 'time_range'
    element_type = [datetime.datetime]
    nullable = True
    input_types = [list,tuple]
    return_type = tuple
    unique = False
    default = None
    
    def validate_all(self,value):
        if len(value) != 2:
            msg = 'There must be two elements in the sequence.'
            raise(DefinitionValidationError(self,msg))
        if value[0] > value[1]:
            msg = 'The second element must be >= the first element.'
            raise(DefinitionValidationError(self,msg))
    
    def _get_meta_(self):
        if self.value == None:
            msg = 'No time range subset.'
        else:
            msg = 'The following time range subset was applied to all request datasets: {0}'.format(self.value)
        return(msg)


class TimeRegion(base.OcgParameter):
    name = 'time_region'
    nullable = True
    default = None
    return_type = dict
    input_types = [dict,OrderedDict]
    
    def _parse_(self,value):
        if value != None:
            ## add missing keys
            for add_key in ['month','year']:
                if add_key not in value:
                    value.update({add_key:None})
            ## confirm only month and year keys are present
            for key in value.keys():
                if key not in ['month','year']:
                    raise(DefinitionValidationError(self,'Time region keys must be month and/or year.'))
            if all([i is None for i in value.values()]):
                value = None
        return(value)
    
    def _get_meta_(self):
        if self.value == None:
            msg = 'No time region subset.'
        else:
            msg = 'The following time region subset was applied to all request datasets: {0}'.format(self.value)
        return(msg)


class VectorWrap(base.BooleanParameter):
    name = 'vector_wrap'
    default = True
    meta_true = 'Geographic coordinates wrapped from -180 to 180 degrees longitude.'
    meta_false = 'Geographic coordinates match the target dataset coordinate wrapping and may be in the range 0 to 360.'
