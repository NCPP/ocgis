class FunctionDefinitionError(Exception):
    """
    Superclass for all exceptions raised from improperly
    configured function definition dictionaries.
    
    f -- dictionary definition for a function.
    """
    
    def __init__(self,f):
        self.f = f


class FunctionNotNamedError(FunctionDefinitionError):
    """
    Raised when a function with parameters is not supplied a name.
    """
        
    def __str__(self):
        msg = ('The dictionary definition for a function having arguments must '
              'have a name.')
        return('{0}\n{1}'.format(msg,self.f))
    
    
class FunctionNameError(FunctionNotNamedError):
    """
    Raised when a function name is improperly formed.
    """

    def __str__(self):
        msg = ('The function name "{0}" must be less than 14 characters in '
               'length, be composed of alphanumeric and/or _ characters, and '
               'must not begin with an integer or _').format(self.f['name'])
        return(msg)