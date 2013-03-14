from ocgis.api.parms import base
from ocgis.exc import DefinitionValidationError, CannotEncodeUrl
from ocgis.api.dataset.request import RequestDataset, RequestDatasetCollection
from ocgis.api.geometry import SelectionGeometry
from ocgis.util.shp_cabinet import ShpCabinet
from shapely.geometry.polygon import Polygon


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
    meta_true = 'Raw values will be used for calculations. These are the original data values linked to a selection value.'
    meta_false = 'Aggregated values will be used during the calculation.'


class Dataset(base.OcgParameter):
    name = 'dataset'
    nullable = False
    default = None
    input_types = [RequestDataset,list,tuple,RequestDatasetCollection]
    return_type = RequestDatasetCollection
    
    def __init__(self,arg):
        if isinstance(arg,RequestDatasetCollection):
            init_value = arg
        else:
            if isinstance(arg,RequestDataset):
                itr = [arg]
            else:
                itr = arg
            rdc = RequestDatasetCollection()
            for rd in itr:
                rdc.update(rd)
            init_value = rdc
        super(Dataset,self).__init__(init_value)
    
    def _get_meta_(self):
        raise(NotImplementedError)
    
    def _get_url_string_(self):
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
        base.OcgParameter.__init__(self,*args,**kwds)
    
    def __repr__(self):
        if len(self.value) == 1 and self.value[0]['geom'] is None:
            value = 'none'
        elif self._shp_key is not None:
            value = self._shp_key
        elif self._bounds is not None:
            value = '|'.join(map(str,self._bounds))
        else:
            value = '<{0} geometry(s)>'.format(len(self.value))
        ret = '{0}={1}'.format(self.name,value)
        return(ret)
    
    def get_url_string(self):
        if len(self.value) > 1:
            raise(CannotEncodeUrl('Too many custom geometries to encode.'))
        else:
            ret = str(self)
            ret = ret.split('=')[1]
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
                if self.select_ugid is None or self.select_ugid.value is None:
                    ret = sc.get_geoms(value)
                else:
                    ret = sc.get_geoms(value,attr_filter={'ugid':self.select_ugid.value})
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
    valid = ('numpy','shp','csv','keyed','meta','nc')
    
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

