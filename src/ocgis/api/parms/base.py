from abc import ABCMeta, abstractproperty
from ocgis.exc import DefinitionValidationError
from copy import deepcopy


class OcgParameter(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def input_types(self): pass
    @abstractproperty
    def name(self): pass
    @abstractproperty
    def nullable(self): pass
    @abstractproperty
    def return_type(self): pass
    
    def __init__(self,init_value=None):
        self._check_types = list(self.input_types) + [basestring,type(None)]
        if init_value is None:
            self.value = self.default
        else:
            self.value = init_value
        
    def __repr__(self):
        ret = '{0}={1}'.format(self.name,self.value)
        return(ret)
    
    @property
    def value(self):
        return(self._value)
    @value.setter
    def value(self,value):
        type_matches = map(lambda x: isinstance(value,x),self._check_types)
        if not any(type_matches):
            raise(DefinitionValidationError(self,'Input value type "{1}" is not in accepted types: {0}'.format(self._check_types,type(value))))
        if isinstance(value,basestring):
            value = self.format_string(value)
        value = deepcopy(value)
        ret = self.format(value)
        self.validate(ret)
        self._value = ret

    def format(self,value):
        ret = self._format_(value)
        return(ret)
        
    def format_string(self,value):
        lowered = value.lower().strip()
        if lowered == 'none':
            ret = None
        else:
            ret = self._format_string_(lowered)
        return(ret)
    
    def get_url_string(self):
        return(str(self._get_url_string_()).lower())
    
    def validate(self,value):
        if value is None:
            if not self.nullable:
                raise(DefinitionValidationError(self,'Argument is not nullable.'))
            else:
                pass
        else:
            try:
                value = self.return_type(value)
            except:
                raise(DefinitionValidationError(self,'Return type does not match.'))
            self._validate_(value)
    
    def _format_(self,value):
        return(value)
        
    def _format_string_(self,value):
        return(value)
    
    def _get_url_string_(self):
        return(self.value)
    
    def _validate_(self,value):
        pass
    
    
class BooleanParameter(OcgParameter):
    nullable = False
    return_type = bool
    input_types = [bool,int]
    
    def _format_(self,value):
        if value == 0:
            ret = False
        elif value == 1:
            ret = True
        else:
            ret = value
        return(ret)
    
    def _format_string_(self,value):
        m = {True:['true','t','1'],
             False:['false','f','0']}
        for k,v in m.iteritems():
            if value in v:
                return(k)
            
    def _get_url_string_(self):
        m = {True:1,False:0}
        return(m[self.value])
    
    
class StringOptionParameter(OcgParameter):
    nullable = False
    return_type = str
    input_types = [str]
    
    @abstractproperty
    def valid(self): pass
    
    def _validate_(self,value):
        if value not in self.valid:
            raise(DefinitionValidationError(self,"Valid arguments are: {0}.".format(self.valid)))
        
        
class IterableParameter(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def element_type(self): pass
    
    def format(self,value):
        if value is None:
            ret = None
        else:
            ret = [OcgParameter.format(self,element) for element in value]
            for idx in range(len(ret)):
                try:
                    ret[idx] = self.element_type(ret[idx])
                except:
                    raise(DefinitionValidationError(self,'Element type incorrect.'))
            ret = self.format_all(ret)
        return(ret)
    
    def format_all(self,value):
        return(value)