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
        msg = ('definition dict validation raised an exception on the argument '
               '"{0}" with the message: "{1}"')
        return(msg.format(self.ocg_argument.name,self.msg))