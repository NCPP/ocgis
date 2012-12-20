class OcgException(Exception):
    pass


class InterpreterException(OcgException):
    pass


class InterpreterNotRecognized(InterpreterException):
    pass


class DefinitionValidationError(OcgException):
    
    def __init__(self,ocg_argument,msg):
        self.ocg_argument = ocg_argument
        self.msg = msg
        
    def __str__(self):
        msg = ('operations validation raised an exception on the argument '
               '"{0}" with the message: "{1}"')
        try:
            msg = msg.format(self.ocg_argument.name,self.msg)
        except AttributeError:
            msg = msg.format(self.ocg_argument._name,self.msg)
        return(msg.format(self.ocg_argument.name,self.msg))
    

class SubsetException(OcgException):
    pass


class MaskedDataError(SubsetException):
    def __str__(self):
        return('Geometric intersection returns all masked values.')
    
    
class ExtentError(SubsetException):
    def __str__(self):
        return('Geometric intersection is empty.')
    
    
class EmptyDataNotAllowed(SubsetException):
    def __str__(self):
        return('Interesection returned empty, but empty data not allowed.')
    
    
class EmptyData(SubsetException):
    def __str__(self):
        return('Empty data returned.')