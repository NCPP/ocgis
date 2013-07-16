#from piston.utils import rc
from django.http import HttpResponse

class rc_factory(object):
    BAD_REQUEST = [400,'Bad Request']
rc = rc_factory()


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
        response = HttpResponse(self._rc[1],
                                status=self._rc[0],
                                content_type='text/plain')
        if self.msg is not None:
            response.write('\n\n'+self.msg)
        return(response)
    
    
class AggregateFunctionError(OcgUrlError):
    _rc = rc.BAD_REQUEST
    _msg = 'Using raw values in statistics functions is only allowed with an aggregated spatial operation.'
    
    
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
    _msg = ("Simulation data matching parameters in the URL were not found. "
            "Ensure the combination of "
            "archive, climate model, emissions scenario, run, and variable "
            "specified in the URL match a record in the simulation output table.")
    
    
class AoiError(OcgUrlError):
    _rc = rc.BAD_REQUEST
    _msg = ('The user-supplied geometry returns an empty intersection with the '
            'data extent of the requested simulation output.')
    
    
class UncaughtRuntimeError(OcgUrlError):
    _rc = rc.BAD_REQUEST
    _msg = ('An uncaught RuntimeError has occurred when attempting to download '
            'the requested data. This may sometimes be solved by resubmitting '
            'the request. If this does not solve the problem, please contact '
            'the system administrator being sure to include the URL raising '
            'the error.')