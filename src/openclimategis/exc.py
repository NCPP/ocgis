from piston.utils import rc


class OcgException(Exception):
    pass


class DatasetExists(OcgException):
    
    def __init__(self,uri):
        self.uri = uri
        
    def __str__(self):
        msg = 'Dataset with URI={0} already exists.'.format(self.uri)
        return(msg)
    
    
class OcgUrlError(OcgException):
    _rc = None
    _msg = None
    
    def __init__(self,msg=None):
        assert(self._rc is not None)
        
        self.msg = msg or self._msg
        
    def __str__(self):
        return(self.msg)
        
    def response(self):
        if self.msg is not None:
            self._rc.write('\n\n'+self.msg)
        return(self._rc)
    
    
class SlugError(OcgUrlError):
    _rc = rc.BAD_REQUEST
    _msg = 'Cannot parse URL argument "{0}" following slug "{1}".'
    
    def __init__(self,slug):
        self.slug = slug
        self.msg = self.format_msg()
        super(SlugError,self).__init__(msg=self.msg)
        
    def format_msg(self):
        return(self._msg.format(self.slug.url_arg,self.slug.code))
        
        
class NoRecordsFound(SlugError):
    _msg = 'No records found for URL argument "{0}" following slug "{1}".'


class MultipleRecordsFound(SlugError):
    _msg = ('Multiple records found for URL argument {0} following slug "{1}".'
           ' Only one record should be returned by this argument.')
    
    
class UserGeometryNotFound(SlugError):
    _msg = ('The requested user geometry "{0}" could not be found.')
    
    def format_msg(self):
        return(self._msg.format(self.slug.url_arg))
    
    
class MalformedSimulationOutputSelection(OcgUrlError):
    _rc = rc.BAD_REQUEST
    _msg = ("This URL does not return a valid simulation output. Ensure the"
            " combination of archive, climate model, scenario, run, and variable"
            " return a valid simulation output.")