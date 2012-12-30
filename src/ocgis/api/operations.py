from definition import * #@UnusedWildImport
from ocgis.exc import DefinitionValidationError
from ocgis.api.interpreter import OcgInterpreter


class OcgOperations(object):
    """Entry point for OCGIS operations. Parameters may be modified before an
    execution. However, the object SHOULD NOT be reused following an execution
    as the software may add/modify attribute contents. Instantiate a new object
    following an execution.
    
    The only required argument is "dataset".
    
    Attributes:
      All keyword arguments are exposed as public attributes which can be
        arbitrarily set using standard syntax:
        
        ops = OcgOperations(dataset={'uri':'/some/dataset','variable':'foo'})
        ops.aggregate = True
        
        The builtins "__getattribute__" and "__setattr__" are overloaded to
        perform validation and formatting upon assignment and to properly return
        parameter values from internal objects.
    """
    
    def __init__(self,dataset=None,spatial_operation=None,geom=None,aggregate=None,
                 time_range=None,level_range=None,calc=None,calc_grouping=None,
                 calc_raw=None,interface=None,snippet=None,backend=None,request_url=None,
                 prefix=None,output_format=None,output_grouping=None,agg_selection=None,
                 select_ugid=None,vector_wrap=None,allow_empty=None):
        
        ## Tells "__setattr__" to not perform global validation until all
        ## values are set initially.
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
        
        ## Initial values have been set and global validation should now occur
        ## when any parameters are updated.
        self._is_init = False
        self._validate_()
        
    def __iter__(self):
        for ii in range(len(self.dataset)):
            ret = {'dataset':self.dataset[ii],
                   'time_range':self.time_range[ii],
                   'level_range':self.level_range[ii]}
            yield(ret)
        
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
    
    def execute(self):
        interp = OcgInterpreter(self)
        return(interp.execute())
    
    def _get_object_(self,name):
        return(object.__getattribute__(self,name))
    
    def _validate_(self):
        for attr in ['time_range','level_range']:
            parm = getattr(self,attr)
            if parm is None:
                parm = [None]
            if len(parm) < len(self.dataset):
                if len(parm) == 1:
                    setattr(self,attr,[parm[0] for ii in range(len(self.dataset))])
                else:
                    raise(DefinitionValidationError(self._get_object_(attr),
                          'range must have length equal to the number of requested datasets or a length of one.'))
    

if __name__ == '__main__':
    import doctest #@Reimport
    doctest.testmod()
