from ocgis.api.parms import base
from ocgis.exc import DefinitionValidationError, CannotEncodeUrl
from ocgis.api.dataset.request import RequestDataset, RequestDatasetCollection
from ocgis.api.geometry import SelectionGeometry
from ocgis.util.shp_cabinet import ShpCabinet
from shapely.geometry.polygon import Polygon
from ocgis.calc.base import OcgFunctionTree
from ocgis.calc import library
from collections import OrderedDict


class Abstraction(base.StringOptionParameter):
    name = 'abstraction'
    default = 'polygon'
    valid = ('point','polygon')
    
    def _get_meta_(self):
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
        if self.value is None:
            fill = None
        else:
            fill = self.get_url_string()
        msg = '{0}={1}'.format(self.name,fill)
        return(msg)
    
    def get_url_string(self):
        elements = []
        for element in self.value:
            strings = []
            template = '{0}~{1}'
            if element['ref'] != library.SampleSize:
                strings.append(template.format(element['func'],element['name']))
                for k,v in element['kwds'].iteritems():
                    strings.append(template.format(k,v))
            if len(strings) > 0:
                elements.append('!'.join(strings))
        ret = '|'.join(elements)
        return(ret)
    
    def parse_all(self,values):
        values.append(self._parse_({'func':'n','name':'n'}))
        return(values)
    
    def _get_meta_(self):
        if self.value is None:
            ret = 'No computations applied.'
        else:
            ret = ['The following computations were applied:']
            for ii in self.value:
                ret.append('{0}: {1}'.format(ii['name'],ii['ref'].description))
        return(ret)
    
    def _parse_(self,value):
        potentials = OcgFunctionTree.get_potentials()
        for p in potentials:
            if p[0] == value['func']:
                value['ref'] = getattr(library,p[1])
                break
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

    
class CalcGrouping(base.IterableParameter,base.OcgParameter):
    name = 'calc_grouping'
    nullable = True
    input_types = [list,tuple]
    return_type = tuple
    default = None
    element_type = str
    unique = True
    
    def _get_meta_(self):
        if self.value is None:
            msg = 'No temporal aggregation applied.'
        else:
            msg = 'Temporal aggregation determined by the following group(s): {0}'.format(self.value)
        return(msg)
    
    def _validate_(self,value):
        for val in value:
            if val not in ['day','month','year','hour','minute','second']:
                raise(DefinitionValidationError(self,'"{0}" is not a valid temporal group.'.format(val)))
            
            
class CalcRaw(base.BooleanParameter):
    name = 'calc_raw'
    default = False
    meta_true = 'Raw values will be used for calculations. These are the original data values linked to a selection geometry.'
    meta_false = 'Aggregated values will be used during the calculation.'


class Dataset(base.OcgParameter):
    name = 'dataset'
    nullable = False
    default = None
    input_types = [RequestDataset,list,tuple,RequestDatasetCollection,dict]
    return_type = RequestDatasetCollection
    
    def __init__(self,arg):
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
    
    def get_url_string(self):
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
    
    def _parse_string_(self,lowered):
        raise(NotImplementedError)


class Geom(base.IterableParameter,base.OcgParameter):
    name = 'geom'
    nullable = True
    default = SelectionGeometry([{'ugid':1,'geom':None}])
    input_types = [SelectionGeometry,list,tuple]
    return_type = SelectionGeometry
    unique = False
    element_type = dict
    _shp_key = None
    _bounds = None
    
    def __init__(self,*args,**kwds):
        self.select_ugid = kwds.pop('select_ugid',None)
        args = [self] + list(args)
        base.OcgParameter.__init__(*args,**kwds)
    
    def __repr__(self):
        if len(self.value) == 1 and self.value[0]['geom'] is None:
            value = None
        elif self._shp_key is not None:
            value = self._shp_key
        elif self._bounds is not None:
            value = '|'.join(map(str,self._bounds))
        else:
            value = '<{0} geometry(s)>'.format(len(self.value))
        ret = '{0}={1}'.format(self.name,value)
        return(ret)
    
    @property
    def is_empty(self):
        if self.value[0]['geom'] is None:
            ret = True
        else:
            ret = False
        return(ret)
    
    def _get_value_(self):
        ret = base.OcgParameter._get_value_(self)
        if ret is None:
            ret = self.default
        return(ret)
    value = property(_get_value_,base.OcgParameter._set_value_)
    
    def get_url_string(self):
        if len(self.value) > 1:
            raise(CannotEncodeUrl('Too many custom geometries to encode.'))
        else:
            ret = str(self)
            ret = ret.split('=')[1]
        return(ret)
    
    def parse(self,value):
        if type(value) in [list,tuple] and len(value) == 4:
            ret = self.parse_string('|'.join(map(str,value)))
        else:
            ret = base.IterableParameter.parse(self,value)
        return(ret)
    
    def parse_string(self,value):
        elements = value.split('|')
        try:
            elements = [float(e) for e in elements]
            minx,miny,maxx,maxy = elements
            geom = Polygon(((minx,miny),
                            (minx,maxy),
                            (maxx,maxy),
                            (maxx,miny)))
            if not geom.is_valid:
                raise(DefinitionValidationError(self,'Parsed geometry is not valid.'))
            ret = [{'ugid':1,'geom':geom}]
            self._bounds = elements
        except ValueError:
            sc = ShpCabinet()
            if value in sc.keys():
                self._shp_key = value
                ## get the select_ugid test value.
                try:
                    test_value = self.select_ugid.value
                except AttributeError:
                    test_value = self.select_ugid
                ## return the geometries
                if test_value is None:
                    ret = sc.get_geoms(value)
                else:
                    ret = sc.get_geoms(value,attr_filter={'ugid':test_value})
        return(ret)
    
    def _get_meta_(self):
        if self.value[0]['geom'] is None:
            ret = 'No user-supplied geometry. All data returned.'
        elif self._shp_key is not None:
            ret = 'The selection geometry "{0}" was used for subsetting.'.format(self._shp_key)
        elif self._bounds is not None:
            ret = 'The bounding box coordinates used for subset are: {0}.'.format(self._bounds)
        else:
            ret = '{0} custom user geometries provided.'.format(len(self.value))
        return(ret)

    
class OutputFormat(base.StringOptionParameter):
    name = 'output_format'
    default = 'numpy'
    valid = ('numpy','shp','csv','keyed','meta','nc','shpidx')
    
    def _get_meta_(self):
        ret = 'The output format is "{0}".'.format(self.value)
        return(ret)
    
    
class Prefix(base.OcgParameter):
    name = 'prefix'
    nullable = False
    default = 'ocgis_output'
    input_types = [str]
    return_type = str
    
    def _get_meta_(self):
        msg = 'Data output given the following prefix: {0}.'.format(self.value)
        return(msg)
    

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
    
    
## determine the iterator mode for the converters
def identify_iterator_mode(ops):
    '''raw,agg,calc,multi'''
    mode = 'raw'
    if ops.calc is not None:
        mode = 'calc'
    ops.mode = mode

