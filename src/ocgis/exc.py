class OcgException(Exception):
    """Base class for all OCGIS exceptions."""
    
    def __init__(self,message=None):
        self.message = message
        
    def __str__(self):
        return(self.message)
    
    
class CalculationException(OcgException):
    
    def __init__(self,function_klass,message=None):
        self.function_klass = function_klass
        OcgException.__init__(self,message=message)
        
    def __str__(self):
        msg = 'The function class "{0}" raised an exception with message: "{1}"'.format(self.function_klass.__name__,
                                                                                        self.message)
        return(msg)
        

class SampleSizeNotImplemented(CalculationException):
    pass


class InterpreterException(OcgException):
    pass


class InterpreterNotRecognized(InterpreterException):
    pass


class EmptyIterationError(OcgException):
    
    def __init__(self,obj):
        self.message = 'Iteration on the object "{0}" requested, but the object is empty.'.format(obj)
        
        
class CFException(OcgException):
    pass


class ProjectionCoordinateNotFound(CFException):
    
    def __init__(self,target):
        self.message = 'The projection coordinate "{0}" was not found in the dataset.'.format(target)
        
        
class ProjectionDoesNotMatch(CFException):
    pass


class DimensionNotFound(CFException):
    pass


class DefinitionValidationError(OcgException):
    """Raised when validation fails on :class:`~ocgis.OcgOperations`.
    
    :param ocg_argument: The origin of the exception.
    :type ocg_argument: :class:`ocgis.api.definition.OcgParameter`, str
    :param msg: The message related to the exception to display in the exception's template.
    :type msg: str
    """
    
    def __init__(self,ocg_argument,msg):
        self.ocg_argument = ocg_argument
        
        fmt = ('OcgOperations validation raised an exception on the argument/operation '
               '"{0}" with the message: {1}')
        try:
            msg = fmt.format(ocg_argument.name,msg)
        except AttributeError:
            try:
                msg = fmt.format(ocg_argument._name,msg)
            except AttributeError:
                msg = fmt.format(ocg_argument,msg)
        
        self.message = msg


class ParameterFormattingError(OcgException):
    pass
    
    
class UniqueIdNotFound(OcgException):
    
    def __init__(self):
        self.message = 'No unique ids found.'
    
    
class DummyDimensionEncountered(OcgException):
    pass


class ResolutionError(OcgException):
    pass
    

class SubsetException(OcgException):
    """Base class for all subset exceptions."""
    pass


class OcgisEnvironmentError(OcgException):
    
    def __init__(self,env_parm,msg):
        self.env_parm = env_parm
        self.msg = msg
        
    def __str__(self):
        new_msg = 'Error when setting the ocgis.env variable {0}. The message is: {1}'.format(self.env_parm.name,self.msg)
        return(new_msg)
    

class SpatialWrappingError(OcgException):
    pass
    
    
class ImproperPolygonBoundsError(OcgException):
    pass


class MaskedDataError(SubsetException):
    
    def __init__(self):
        self.message = 'Geometric intersection returned all masked values.'
    
    
class ExtentError(SubsetException):
    
    def __init__(self,message=None):
        self.message = message or 'Geometry intersection is empty.'
        
        
class TemporalExtentError(SubsetException):
    
    def __init__(self):
        self.message = 'Temporal subset returned empty.'
    
    
class EmptyDataNotAllowed(SubsetException):
    """Raised when the empty set for a geometry is returned and `allow_empty`_ is `False`."""
    
    def __init__(self):
        self.message = 'Intersection returned empty, but empty data not allowed.'
    
    
class EmptyData(SubsetException):
    
    def __init__(self,message=None,origin=None):
        self.message = message or 'Empty data returned.'
        self.origin = origin


class EmptySubsetError(SubsetException):
    
    def __init__(self,origin=None):
        self.origin = origin
        
    def __str__(self):
        msg = 'A subset operation on dimension "{0}" returned empty.'.format(self.origin)
        return(msg)
    
    
class NoUnitsError(OcgException):
    '''
    Raised when a :class:`cfunits.Units` object is constructed from a NoneType
    value.
    '''
    
    def __str__(self):
        if self.message is None:
            msg = 'Variable has not been assigned units. Set the "units" attribute to continue.'
        else:
            msg = self.message
        return(msg)


class IncompleteSeasonError(OcgException):
            
    def __init__(self,season,year=None,month=None):
        self.season = season
        self.year = year
        self.month = month
        
    def __str__(self):
        if self.year is not None:
            msg = 'The season specification "{0}" is missing the year "{1}".'.format(self.season,self.year+1)
        if self.month is not None:
            msg = 'The season specification "{0}" is missing the month "{1}".'.format(self.season,self.month)
        return(msg)
