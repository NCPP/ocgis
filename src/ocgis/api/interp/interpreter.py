from ocgis import exc

## TODO: add method to estimate request size

class Interpreter(object):
    '''Superclass for custom interpreter frameworks.
    
    desc :: dict :: Operational arguments for the interpreter to execute.'''
    
    def __init__(self,desc):
        self.desc = desc
        
    @classmethod
    def get_interpreter(cls,desc):
        '''Select interpreter class.'''
        
        from ocgis.api.interp.iocg.interpreter_ocg import OcgInterpreter
        
        imap = {'ocg':OcgInterpreter}
        try:
            return(imap[desc['backend']](desc))
        except KeyError:
            raise(exc.InterpreterNotRecognized)
        
    def check(self):
        '''Validate operation definition dictionary.'''
        raise(NotImplementedError)
    
    def execute(self):
        '''Run requested operations and return a path to the output file or a
        NumPy-based output object depending on specification.'''
        raise(NotImplementedError)
