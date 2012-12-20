from definition import * #@UnusedWildImport
from ocgis.exc import DefinitionValidationError


class OcgOperations(object):
    """
    >>> dataset = [{'uri':'some.path','variable':'foo'}]
    >>> oo = OcgOperations(dataset=[{'uri':'some.path','variable':'foo'}],spatial_operation='clip')
    >>> oo.spatial_operation
    'clip'
    >>> oo.spatial_operation = 'intersects'
    >>> oo.as_dict()
    {'calc_grouping': ['year', 'day', 'month'], 'level_range': None, 'calc_raw': False, 'agg_selection': False, 'output_format': 'keyed', 'spatial_operation': 'intersects', 'dataset': [{'variable': 'foo', 'uri': 'some.path'}], 'snippet': False, 'aggregate': True, 'prefix': None, 'time_range': None, 'geom': [{'geom': None, 'id': 1}], 'interface': None, 'request_url': None, 'calc': None, 'backend': 'ocg'}
    >>> kwds = {'dataset':dataset,'spatial_operation':'clip','aggregate':False}
    >>> oo = OcgOperations(**kwds)
    >>> oo.spatial_operation,oo.aggregate
    ('clip', False)
    """
    
    def __init__(self,dataset=None,spatial_operation=None,geom=None,aggregate=None,
                 time_range=None,level_range=None,calc=None,calc_grouping=None,
                 calc_raw=None,interface=None,snippet=None,backend=None,request_url=None,
                 prefix=None,output_format=None,output_grouping=None,agg_selection=None,
                 select_ugid=None,vector_wrap=None,allow_empty=None):
        
        self._is_init = True
        
        self.dataset = Dataset(dataset)
        self.spatial_operation = SpatialOperation(spatial_operation)
        self.geom = Geom(geom)
        self.aggregate = Aggregate(aggregate)
        self.time_range = TimeRange(time_range)
        self.level_range = LevelRange(level_range)
        self.calc = Calc(calc)
        self.calc_grouping = CalcGrouping(calc_grouping)
        self.calc_raw = CalcRaw(calc_raw)
        self.interface = Interface(interface)
        self.snippet = Snippet(snippet)
        self.backend = Backend(backend)
        self.request_url = RequestUrl(request_url)
        self.prefix = Prefix(prefix)
        self.output_format = OutputFormat(output_format)
        self.output_grouping = output_grouping
        self.agg_selection = AggregateSelection(agg_selection)
        self.select_ugid = SelectUgid(select_ugid)
        self.vector_wrap = VectorWrap(vector_wrap)
        self.allow_empty = AllowEmpty(allow_empty)
        
        self._is_init = False
        self._validate_()
        
    def __repr__(self):
        msg = ['<{0}>:'.format(self.__class__.__name__)]
        for key,value in self.as_dict().iteritems():
            if key == 'geom' and len(value) > 1:
                value = '{0} geometries...'.format(len(value))
            msg.append(' {0}={1}'.format(key,value))
        msg = '\n'.join(msg)
        return(msg)
            
    def __getattribute__(self,name):
        attr = object.__getattribute__(self,name)
        if isinstance(attr,OcgParameter):
            ret = attr.value
        else:
            ret = attr
        return(ret)
    
    def __setattr__(self,name,value):
        if isinstance(value,OcgParameter):
            object.__setattr__(self,name,value)
        else:
            try:
                attr = object.__getattribute__(self,name)
                attr.value = value
            except AttributeError:
                object.__setattr__(self,name,value)
        if self._is_init is False:
            self._validate_()
        
    def as_dict(self):
        ret = {}
        for value in self.__dict__.itervalues():
            try:
                ret.update({value.name:value.value})
            except AttributeError:
                pass
        return(ret)
    
    def _get_object_(self,name):
        return(object.__getattribute__(self,name))
    
    def _validate_(self):
        for attr in ['time_range','level_range']:
            parm = getattr(self,attr)
            if len(parm) < len(self.dataset):
                if len(parm) == 1:
                    setattr(self,attr,[parm[0] for ii in range(len(self.dataset))])
                else:
                    raise(DefinitionValidationError(self._get_object_(attr),
                          'range must have length equal to the number of requested datasets or a length of one.'))
    
    
if __name__ == '__main__':
    import doctest #@Reimport
    doctest.testmod()
