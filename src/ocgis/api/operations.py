from definition import *


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
                 prefix=None,output_format=None,output_grouping=None,agg_selection=None):
        
        self._kwds = locals()
        self.dataset = Dataset
        self.spatial_operation = SpatialOperation
        self.geom = Geom
        self.aggregate = Aggregate
        self.time_range = TimeRange
        self.level_range = LevelRange
        self.calc = Calc
        self.calc_grouping = CalcGrouping
        self.calc_raw = CalcRaw
        self.interface = Interface
        self.snippet = Snippet
        self.backend = Backend
        self.request_url = RequestUrl
        self.prefix = Prefix
        self.output_format = OutputFormat
        self.output_grouping = output_grouping
        self.agg_selection = AggregateSelection
            
    def __getattribute__(self,name):
        attr = object.__getattribute__(self,name)
        if isinstance(attr,OcgParameter):
            ret = attr.value
        else:
            ret = attr
        return(ret)
    
    def __setattr__(self,name,value):
        try:
            if issubclass(value,OcgParameter):
                value = value()
                value.value = self._kwds.get(value.name)
                object.__setattr__(self,name,value)
            else:
                raise(NotImplementedError)
        except TypeError:
            try:
                attr = object.__getattribute__(self,name)
                attr.value = value
            except AttributeError:
                object.__setattr__(self,name,value)
        
    def as_dict(self):
        ret = {}
        for value in self.__dict__.itervalues():
            try:
                ret.update({value.name:value.value})
            except AttributeError:
                pass
        return(ret)
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
