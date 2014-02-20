from abc import ABCMeta, abstractproperty, abstractmethod
from ocgis.exc import DefinitionValidationError
from copy import deepcopy
from types import NoneType


class OcgParameter(object):
    '''Abstract base class for input parameters.
    
    :type init_value: Varies depending on overloaded parameter class.
    '''
    __metaclass__ = ABCMeta
    _lower_string = True #: if set to False, do not lower input strings
    _perform_deepcopy = True #: if False, do not perform deepcopy operation on value set
    
    def __init__(self,init_value=None):
        if init_value is None:
            self.value = self.default
        else:
            self.value = init_value
        
    def __str__(self):
        ret = '{0}={1}'.format(self.name,self.value)
        return(ret)
    
    @abstractproperty
    def input_types(self):
        '''List of acceptable input types - `basestring` is always appended to
        this list.'''
        [type]
        
    @abstractproperty
    def name(self):
        '''Name of the input parameter.'''
        str
        
    @abstractproperty
    def nullable(self):
        '''If True, the parameter is nullable.'''
        bool
        
    ##TODO: switch name to "return_types"
    @abstractproperty
    def return_type(self):
        '''List of acceptable return types.'''
        [type]
    
    def _get_value_(self):
        return(self._value)
    def _set_value_(self,value):
        input_types = self.input_types + [basestring,NoneType]
        type_matches = map(lambda x: isinstance(value,x),input_types)
        if not any(type_matches):
            raise(DefinitionValidationError(self,'Input value type "{1}" is not in accepted types: {0}'.format(input_types,type(value))))
        if isinstance(value,basestring):
            value = self.parse_string(value)
        else:
            if self._perform_deepcopy:
                value = deepcopy(value)
        ret = self.parse(value)
        try:
            if ret is not None:
                try:
                    if self.return_type != type(ret):
                        ret = self.return_type(ret)
                except:
                    if type(ret) not in self.return_type:
                        if not any([isinstance(value,rt) for rt in self.return_type]):
                            raise
        except:
            raise(DefinitionValidationError(self,'Return type does not match.'))
        self.validate(ret)
        self._value = ret
        ## final hook for any modifications to the object
        self.finalize()
    ## Return the value associated with the paramter. Note it does not return
    ## the object.
    value = property(_get_value_,_set_value_)
    
    def finalize(self):
        pass
        
    def get_meta(self):
        ''':rtype: list of strings'''
        subrows = self._get_meta_()
        if isinstance(subrows,basestring):
            subrows = [subrows]
        rows = ['* '+str(self)]
        rows.extend(subrows)
        rows.append('')
        return(rows)
    
    def parse(self,value):
        ''':rtype: varies depending on `ocgis.api.parms.base.OcgParamter` subclass'''
        ret = self._parse_(value)
        return(ret)
        
    def parse_string(self,value):
        ''':rtype: varies depending on `ocgis.api.parms.base.OcgParamter` subclass'''
        if self._lower_string:
            modified = value.lower()
        else:
            modified = value
        modified = modified.strip()
        if modified == 'none':
            ret = None
        else:
            ret = self._parse_string_(modified)
        return(ret)
    
    def validate(self,value):
        ''':rtype: void'''
        if value is None:
            if not self.nullable:
                raise(DefinitionValidationError(self,'Argument is not nullable.'))
            else:
                pass
        else:
            self._validate_(value)
            
    @abstractmethod
    def _get_meta_(self):
        return(list)
    
    def _parse_(self,value):
        return(value)
        
    def _parse_string_(self,value):
        return(value)
    
    def _validate_(self,value):
        pass
    
    
class BooleanParameter(OcgParameter):
    nullable = False
    return_type = bool
    input_types = [bool,int]
    
    @abstractproperty
    def meta_true(self): str
    @abstractproperty
    def meta_false(self): str
    
    def _get_meta_(self):
        if self.value:
            ret = self.meta_true
        else:
            ret = self.meta_false
        return(ret)
    
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
            
            
class StringParameter(OcgParameter):
    __metaclass__ = ABCMeta
    
    return_type = str
    input_types = [str]
    
    def __str__(self):
        ret = '{0}="{1}"'.format(self.name,self.value)
        return(ret)
    
    
class StringOptionParameter(StringParameter):
    __metaclass__ = ABCMeta
    
    nullable = False
    
    @abstractproperty
    def valid(self): [str]
    
    def _validate_(self,value):
        if value not in self.valid:
            raise(DefinitionValidationError(self,"Valid arguments are: {0}.".format(self.valid)))
        
        
class IterableParameter(object):
    __metaclass__ = ABCMeta
    
    split_string = '|'
    
    @abstractproperty
    def element_type(self): type
    @abstractproperty
    def unique(self): bool
    
    def parse(self,value,check_basestrings=True):
        if value is None:
            ret = None
        else:
            try:
                itr = iter(value)
            except TypeError:
                itr = iter([value])
            ret = [OcgParameter.parse(self,element) for element in itr]
            if self.unique:
                try:
                    if len(set(ret)) < len(value):
                        raise(DefinitionValidationError(self,'Argument sequence must have unique elements.'))
                ## elements may not be reduceable to a set. attempt to reduce
                ## the individual elements and compare by this method.
                except TypeError:
                    for start_idx,element in enumerate(value):
                        ## some element sequences may have string flags which should
                        ## be ignored.
                        if check_basestrings == False:
                            if isinstance(element,basestring):
                                continue
                        element_set = set(element)
                        ## individual element sequences must be unique (i.e. [1,1,2] is not acceptable)
                        if len(element_set) != len(element):
                            raise(DefinitionValidationError(self,'Argument element sequences must be composed of unique values.'))
                        ## this compares the uniqueness of an element in the sequence
                        ## to other components in the sequence.
                        for to_check in range(start_idx+1,len(value)):
                            if len(element_set.intersection(set(value[to_check]))) > 0:
                                raise(DefinitionValidationError(self,'Argument sequence must have unique elements.'))
            for idx in range(len(ret)):
                try:
                    ret[idx] = self.element_type(ret[idx])
                except TypeError:
                    if type(ret[idx]) not in self.element_type:
                        raise(DefinitionValidationError(self,'Element type incorrect. Acceptable types are: {0}'.format(self.element_type)))
                    else:
                        pass
                except ValueError:
                    raise(DefinitionValidationError(self,'Element type incorrect. Acceptable types are: {0}'.format(self.element_type)))
            ret = self.parse_all(ret)
            self.validate_all(ret)
        return(ret)
    
    def parse_all(self,values):
        return(values)
    
    def parse_string(self,value):
        ret = value.split(self.split_string)
        ret = [OcgParameter.parse_string(self,element) for element in ret]
        if ret == [None]:
            ret = None
        return(ret)
    
    def validate_all(self,values):
        pass
    
    def element_to_string(self,element):
        return(str(element))
