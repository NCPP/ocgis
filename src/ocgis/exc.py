class OcgException(Exception):
    """Base class for all OCGIS exceptions."""
    
    def __init__(self,message=None):
        self.message = message
        
    def __str__(self):
        return(self.message)


class InterpreterException(OcgException):
    pass


class InterpreterNotRecognized(InterpreterException):
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
        
        fmt = ('Operations validation raised an exception on the argument or operation '
               '"{0}" with the message: "{1}"')
        try:
            msg = fmt.format(ocg_argument.name,msg)
        except AttributeError:
            try:
                msg = fmt.format(ocg_argument._name,msg)
            except AttributeError:
                msg = fmt.format(ocg_argument,msg)
        
        self.message = msg


class CannotEncodeUrl(OcgException):
    """Raised when a URL may not be encoded from an :func:`~ocgis.OcgOperations.as_qs` call."""
    pass


class ParameterFormattingError(OcgException):
    pass
    
    
class UniqueIdNotFound(OcgException):
    
    def __init__(self):
        self.message = 'No unique ids found.'
    
    
class DummyDimensionEncountered(OcgException):
    pass
    

class SubsetException(OcgException):
    """Base class for all subset exceptions."""
    pass


class MaskedDataError(SubsetException):
    
    def __init__(self):
        self.message = 'Geometric intersection returned all masked values.'
    
    
class ExtentError(SubsetException):
    
    def __init__(self):
        self.message = 'Geometry intersection is empty.'
    
    
class EmptyDataNotAllowed(SubsetException):
    """Raised when the empty set for a geometry is returned and `allow_empty`_ is `False`."""
    
    def __init__(self):
        self.message = 'Intersection returned empty, but empty data not allowed.'
    
    
class EmptyData(SubsetException):
    
    def __init__(self,message=None):
        self.message = message or 'Empty data returned.'
