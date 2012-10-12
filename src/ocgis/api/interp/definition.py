import itertools
import datetime
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from copy import copy
from ocgis.exc import DefinitionValidationError
from ocgis.calc.wrap.base import OcgFunctionTree, OcgCvArgFunction
from ocgis.calc.wrap import library


class OcgArgument(object):
    name = None
    can_be_none = None
    
    def __init__(self,value):
        self._assert_(self.name is not None,'name must be provided')
        self._assert_(self.can_be_none is not None,'specify nullability')
        if value is None and not self.can_be_none:
            msg = 'no value provided but argument cannot be None'
            raise(DefinitionValidationError(self,msg))
        elif value is None and self.can_be_none:
            try:
                self.value = self._none_format_()
            except NotImplementedError:
                self.value = value
        else:
            self.value = self.format(value)
            self.validate()
        
    def format(self,value):
        return(value)
    
    def _none_format_(self):
        raise(NotImplementedError)
        
    def validate(self):
        raise(NotImplementedError)
    
    def _assert_(self,test,errmsg):
        try:
            assert(test)
        except AssertionError:
            raise(DefinitionValidationError(self,errmsg))
        
    def message(self):
        raise(NotImplementedError)
    
    
class BooleanArgument(OcgArgument):
    
    def validate(self):
        assert(type(self.value) == bool)
        
        
class StringArgument(OcgArgument):
    _possible = []
    
    def __init__(self,*args,**kwds):
        msg = 'argument possibilities not provided'
        self._assert_(len(self._possible) > 0,msg)
        super(StringArgument,self).__init__(*args,**kwds)
    
    def format(self,value):
        return(value.lower())
    
    def validate(self):
        msg = '"{0}" not in found in "{1}"'.format(self.value,self._possible)
        self._assert_(self.value in self._possible,msg)
        
        
class StringListArgument(StringArgument):

    def format(self,value):
        try:
            iterator = iter(value)
        except TypeError:
            iterator = iter([value])
        ret = [ii.value for ii in iterator]
        return(ret)
    
    def validate(self):
        valid = False
        for p,v in itertools.product(self._possible,[self.value]):
            if set(p) == set(v):
                valid = True
                break
        if not valid:
            msg = '"{0}" is not a supported grouping'.format(self.value)
            raise(DefinitionValidationError(self,msg))


class Backend(StringArgument):

    _possible = ['ocg']
    can_be_none = False
    name = 'backend'
    
    def message(self):
        if self.value == 'ocg':
            msg = ('OpenClimateGIS used as geoprocessing and calculation '
                   'backend.')
        else:
            raise(NotImplementedError)
        return(msg)


class Grouping(StringArgument):
    _possible = ['day','month','year']
    can_be_none = True
    name = 'grouping_element'


class CalcGrouping(StringListArgument):
    '''
    >>> cg = CalcGrouping(Grouping('day'))
    >>> cg.value
    ['day']
    >>> cg = CalcGrouping([Grouping('month'),Grouping('day')])
    >>> cg.value
    ['month', 'day']
    '''
    _possible = [['day','month','year'],
                 ['month'],
                 ['year'],
                 ['day'],
                 ['month','year'],
                 ['day','month'],
                 ['day','year']]
    can_be_none = True
    name = 'calc_grouping'
    
    def __init__(self,value):
        if value is not None:
            if type(value) not in [list,tuple]:
                value = [value]
            value = [Grouping(v) for v in value]
        super(CalcGrouping,self).__init__(value)
        
    def message(self):
        msg = ('Temporal aggregation determined by the following group(s): {0}')
        msg = msg.format(self.value)
        return(msg)
    
    
class LevelRange(OcgArgument):
    '''
    >>> LevelRange([1,2]).value
    [1, 2]
    >>> LevelRange([1,2,3,]).value # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DefinitionValidationError:
    >>> LevelRange(1).value
    [1]
    '''
    can_be_none = True
    name = 'level_range'
    
    def format(self,value):
        if type(value) not in [list,tuple]:
            value = [value]
        return(value)
    
    def validate(self):
        for ii in self.value:
            self._assert_(type(ii) == int,'argument must be an integer')
        self._assert_(len(self.value) in [1,2],'only arguments of length 1 or 2')

    def message(self):
        if self.value is None:
            msg = ('No level range provided. If variable(s) has/have a level dimesion,'
                   ' all levels will be returned.')
        else:
            msg = 'Level range returned is {0}.'.format(self.value)
        return(msg)
        
class OutputFormat(StringArgument):
    _possible = ['numpy','shpidx','shp','csv','nc','keyed','meta']
    name = 'output_format'
    can_be_none = False
    
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
    
    
class SpatialOperation(StringArgument):
    _possible = ['intersects','clip']
    name = 'spatial_operation'
    can_be_none = False
    
    def message(self):
        if self.value == 'intersects':
            msg = 'The "intersects" operation returns all grid cells overlapping the selection geometry. Note that this does NOT include those geometries touching the selection geometry. In the case of point data, only those points occurring inside the selection geometry are returned.'
        if self.value == 'clip':
            msg = 'The "clip" operation is a full geometric intersection of the selection and target geometries.'
        return(msg)
    
class CalcRaw(BooleanArgument):
    name = 'calc_raw'
    can_be_none = True
    
    def message(self):
        if self.value:
            msg = 'Raw values will be used for calculations. These are the original data values linked to a selection value.'
        else:
            msg = 'Aggregated values will be used during the calculation.'
        return(msg)
    
    
class Aggregate(BooleanArgument):
    name = 'aggregate'
    can_be_none = False
    
    def message(self):
        if self.value:
            msg = 'Selected geometries are aggregated (unioned), and associated data values are area-weighted based on final area following the spatial operation. Weights are normalized using the maximum area of the geometry set.'
        else:
            msg = 'Selected geometries are not aggregated (unioned).'
        return(msg)
    
class TimeRange(OcgArgument):
    name = 'time_range'
    can_be_none = True
    
    def validate(self):
        for ii in self.value:
            self._assert_(type(ii) == datetime.datetime,
             'must be datetime.datetime.object')
        self._assert_(len(self.value) == 2,'must be of length 2')
        self._assert_(self.value[0] <= self.value[1],('first element must be'
                                                      ' <= second element'))
    def message(self):
        if self.value is None:
            msg = 'Time selection range not provided. All time points returned.'
        else:
            msg = 'Time selection range is: {0}. Selection is inclusive.'.format(self.value)
        return(msg)
        
class Calc(OcgArgument):
    name = 'calc'
    can_be_none = True
    
    def format(self,value):
        funcs_copy = copy(value)
        if not any([ii['func'] == 'n' for ii in value]):
            funcs_copy.insert(0,{'func':'n'})
        
        potentials = OcgFunctionTree.get_potentials()
        for f in funcs_copy:
            for p in potentials:
                if p[0] == f['func']:
                    f['ref'] = getattr(library,p[1])
                    break
            if 'name' not in f:
                f['name'] = f['func']
            if 'kwds' not in f:
                f['kwds'] = {}
        return(funcs_copy)
    
    def validate(self):
        for ii in self.value:
            self._assert_('func' in ii,('at least the function name is '
                                        'required using the key "func"'))
            
    def message(self):
        if self.value is None:
            msg = 'No calculations requested.'
        else:
            msg = ''
            for ii in self.value:
                msg += '{0} :: {1}\n'.format(ii['name'],ii['ref'].description)
                if len(ii['kwds']) > 0:
                    msg += ' Parameters:\n'
                    for key,value in ii['kwds'].iteritems():
                        msg += '  {0}={1}\n'.format(key,value)
                msg += '\n'
        return(msg)
            
            
class Geom(OcgArgument):
    name = 'geom'
    can_be_none = True
    
    def format(self,value):
        if type(value) not in [list,tuple]:
            value = [value]
        return(value)
    
    def _none_format_(self):
        return([{'id':1,'geom':None}])
    
    def validate(self):
        for ii in self.value:
            self._assert_(type(ii) == dict,
             'list elements must be dictionaries with keys "id" and "geom"')
            self._assert_('id' in ii,'a geom geom must have an id key')
            self._assert_('geom' in ii,'a geom dict must have a geom key')
            self._assert_(type(ii['geom']) in [Polygon,MultiPolygon],
                          'geometry type not recognized')
    
    def message(self):
        for ii in self.value:
            if ii['geom'] is None:
                msg = 'No user-supplied geometry. All data returned.'
                return(msg)
            else:
                msg = '{0} user geometries provided.'.format(len(self.value))
                return(msg)
         
            
class Meta(OcgArgument):
    name = 'meta'
    can_be_none = False
    
    def format(self,value):
        if type(value) not in [list,tuple]:
            value = [value]
        return(value)
    
    def validate(self):
        for ii in self.value:
            self._assert_(type(ii) == dict,'meta list elements must be dicts')
            self._assert_('uri' in ii,'a uri must be provided')
            self._assert_('variable' in ii,'a variable must be provided')
    
    def message(self):
        lines = []
        for ii in self.value:
            lines.append('The variable "{0}" requested from dataset with URI "{1}".'.format(ii['variable'],ii['uri']))
        return('\n'.join(lines))
            
## collection of arguments that comprise an operational call to OCG
DEF_ARGS = [Meta,Backend,TimeRange,LevelRange,Geom,OutputFormat,SpatialOperation,
            Aggregate,CalcRaw,CalcGrouping,Calc]

## dictionary validation and formatting
def validate_update_definition(desc):
    for Da in DEF_ARGS:
        obj = Da(desc.get(Da.name))
        desc.update({Da.name:obj.value})
        
## determine the iterator mode for the converters
def identify_iterator_mode(desc):
    '''raw,agg,calc,multi'''
    mode = 'raw'
    if desc['aggregate']:
        mode = 'agg'
    if desc['calc'] is not None:
        mode = 'calc'
        for f in desc['calc']:
            if isinstance(f['ref'],OcgCvArgFunction):
                mode = 'multi'
                break
    desc['mode'] = mode

            
if __name__ == "__main__":
    import doctest
    doctest.testmod()