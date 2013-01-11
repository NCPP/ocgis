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
    
    def __init__(self,dataset=None,spatial_operation='intersects',geom=None,aggregate=False,
                 time_range=None,level_range=None,calc=None,calc_grouping=None,
                 calc_raw=False,interface={},snippet=False,backend='ocg',request_url=None,
                 prefix=None,output_format='numpy',output_grouping=None,agg_selection=False,
                 select_ugid=None,vector_wrap=True,allow_empty=False):
        """The only required argument is "dataset". All others are provided
        defaults.
        
        Args:
          dataset: Sequence of dictionaries having keys "uri" and "variable". It
            is possible to request data from multiple locations. Multiple time
            and level ranges may also be specified (see their docstrings). For
            example:
            
            Request for a single local dataset:
            dataset=[{'uri':'/some/local/dataset','variable':'foo'}]
            
            Request for a local and remote dataset:
            dataset=[{'uri':'/some/local/dataset','variable':'foo'},
                     {'uri':'http://some.opendap.dataset','variable':'foo2'}]
                     
          snippet: If True, it will return the first time point or time group
            (in the case of calculations) for each requested dataset. Best to
            set to True for any initial request or data inspections.
            
          output_format: String indicating the desired output format. Available
            options are: 'keyed' (default), 'nc', 'shp', or 'csv'.
                     
          time_range: Sequence of Python datetime object sequences. The default 
            returns all time points. If only a single range is provided but 
            multiple datasets are requested, the range is repeated for the 
            length of the dataset sequence. For example:
            
            time_range=[[lower,upper]]
            time_range=[[lower_dataset1,upper_dataset1],
                        [lower_dataset2,upper_dataset2]]
                        
          level_range: Sequence of integer sequences. The same mapping applies
            as "time_range". A value of 1 represents and index of 0 in the
            value array index (i.e. the first returned level). For example:
            
            level_range=[[1,1]]
            level_range=[[1,1],[10,10]]
        
          geom: Sequence of dictionaries composed of an identifier "ugid" and a
            Shapely Polygon/MultiPolygon geometry object "geom". Geometries 
            should always be provided with a WGS84 geographic coordinate system 
            on a -180 to 180 longitudinal spatial domain. The default None will 
            return the entire spatial domain. For example:
            
            geom=[{'ugid':25,'geom':<Shapely Polygon object>},
                  {'ugid':26,'geom':<Shapely MultiPolygon object>}]
        
          vector_wrap: Set to False to return vector geometries in the dataset's
            native longitudinal spatial domain (e.g. -180 to 180 or 0 to 360).
            If True, the default, vector outputs will always be returned on the
            -180 to 180 longitudinal domain.
            
          spatial_operation: There are two options: 'intersects' or 'clip'.
          
          aggregate: If True, geometries coincident with a selection geometry
            are aggregated and the associated values area-weighted to single
            value. Raw values are maintained.
            
          allow_empty: If True, geometric operations returning no data or all
            masked values will be written as empty returns. 
            
          agg_selection: If True, selection geometries are aggregated to a
            single geometry. This is automatically set to True if the requested
            output format is 'nc'.
        """
        
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
        
        def _get_(idx,attr):
            if attr is None:
                ret = None
            else:
                ret = attr[idx]
            return(ret)
        
        for ii in range(len(self.dataset)):
            ret = {'dataset':self.dataset[ii],
                   'time_range':_get_(ii,self.time_range),
                   'level_range':_get_(ii,self.level_range)}
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
            
    def as_url(self,slug=''):
        parts = []
        for key,value in self.__dict__.iteritems():
            if key in ['request_url']:
                continue
            if isinstance(value,OcgParameter):
                parts.append(str(value))
        import ipdb;ipdb.set_trace()
        
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
