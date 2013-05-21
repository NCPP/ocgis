class OcgException(Exception):
    """Base class for all OCGIS exceptions."""
    
    def __init__(self,msg=None):
        self.msg = msg


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
        self.msg = msg
        
    def __str__(self):
        msg = ('Operations validation raised an exception on the argument or operation '
               '"{0}" with the message: "{1}"')
        try:
            msg = msg.format(self.ocg_argument.name,self.msg)
        except AttributeError:
            try:
                msg = msg.format(self.ocg_argument._name,self.msg)
            except AttributeError:
                msg = msg.format(self.ocg_argument,self.msg)
        return(msg)


class CannotEncodeUrl(OcgException):
    """Raised when a URL may not be encoded from an :func:`~ocgis.OcgOperations.as_qs` call."""
    pass


class ParameterFormattingError(OcgException):
    pass
    
    
class UniqueIdNotFound(OcgException):
    
    def __str__(self):
        return('No unique ids found.')
    
    
class DummyDimensionEncountered(OcgException):
    pass
    

class SubsetException(OcgException):
    """Base class for all subset exceptions."""
    pass


class MaskedDataError(SubsetException):
    def __str__(self):
        return('Geometric intersection returns all masked values.')
    
    
class ExtentError(SubsetException):
    def __str__(self):
        return('Geometric intersection is empty. {0}'.format(self.msg))
    
    
class EmptyDataNotAllowed(SubsetException):
    """Raised when the empty set for a geometry is returned and `allow_empty`_ is `False`."""
    def __str__(self):
        return('Intersection returned empty, but empty data not allowed.')
    
    
class EmptyData(SubsetException):
    def __str__(self):
        return('Empty data returned.')