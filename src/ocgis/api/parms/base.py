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
        ret = '{0}={1}'.parse(self.name,self.value)
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
            value = self.parse_string(value)
        else:
            value = deepcopy(value)
        ret = self.parse(value)
        try:
            if ret is not None:
                ret = self.return_type(ret)
        except:
            raise(DefinitionValidationError(self,'Return type does not match.'))
        self.validate(ret)
        self._value = ret

    def parse(self,value):
        ret = self._parse_(value)
        return(ret)
        
    def parse_string(self,value):
        lowered = value.lower().strip()
        if lowered == 'none':
            ret = None
        else:
            ret = self._parse_string_(lowered)
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
            self._validate_(value)
    
    def _parse_(self,value):
        return(value)
        
    def _parse_string_(self,value):
        return(value)
    
    def _get_url_string_(self):
        return(self.value)
    
    def _validate_(self,value):
        pass
    
    
class BooleanParameter(OcgParameter):
    nullable = False
    return_type = bool
    input_types = [bool,int]
    
    def _parse_(self,value):
        if value == 0:
            ret = False
        elif value == 1:
            ret = True
        else:
            ret = value
        return(ret)
    
    def _parse_string_(self,value):
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
    @abstractproperty
    def unique(self): pass
    
    def parse(self,value):
        if value is None:
            ret = None
        else:
            ret = [OcgParameter.parse(self,element) for element in value]
            if self.unique:
                if len(set(ret)) < len(value):
                    raise(DefinitionValidationError(self,'Argument sequence must have unique elements.'))
            for idx in range(len(ret)):
                try:
                    ret[idx] = self.element_type(ret[idx])
                except:
                    raise(DefinitionValidationError(self,'Element type incorrect.'))
            ret = self.parse_all(ret)
        return(ret)
    
    def parse_all(self,value):
        return(value)
    
    def parse_string(self,value):
        ret = value.split(',')
        ret = [OcgParameter.parse_string(self,element) for element in ret]
        if ret == [None]:
            ret = None
        return(ret)

    def get_url_string(self):
        ret = ','.join([self.element_to_string(element) for element in self.value]).lower()
        return(ret)
    
    def element_to_string(self,element):
        return(str(element))