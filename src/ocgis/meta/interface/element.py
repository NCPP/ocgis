from warnings import warn


class ElementNotFound(Exception):
    
    def __init__(self,klass):
        self.klass = klass
        
    def __str__(self):
        args = (self.klass.__name__,
                self.klass._names)
        msg = 'Element model "{0}" found no matches for {1}'.format(*args)
        return(msg)
    
    
class Element(object):
    _names = []
    _ocg_name = None
    _default = None
    
    def __init__(self,dataset,name=None):
        assert(len(self._names) > 0)
        assert(self._ocg_name is not None)
        self.name = name or self._get_name_(dataset)
    
    def calculate(self):
        raise(NotImplementedError)
    
    def _possible_(self,dataset):
        raise(NotImplementedError)
    
    def _get_name_(self,dataset):
        possible = [str(p) for p in self._possible_(dataset)]
        for ii,poss in enumerate(possible,start=1):
            a = [n == poss for n in self._names]
            if any(a):
                ret = self._names[a.index(True)]
                break
            if ii == len(possible):
                if self.__class__._default is None:
                    raise(ElementNotFound(self.__class__))
                else:
                    msg = ('Default value of "{0}" used for "{1}"'.\
                           format(self.__class__._default,
                                  self.__class__.__name__))
                    warn(msg)
                    ret = self.__class__._default
        return(ret)
    
    
class VariableElement(Element):
    _AttributeElements = []
    
    def __init__(self,dataset):
        super(VariableElement,self).__init__(dataset)
        self.dtype = self._get_dtype_(dataset)
        
        for Attr in self._AttributeElements:
            attr = Attr(dataset,self)
            setattr(self,attr._ocg_name,attr)
            
        self.value = self._get_value_(dataset)
    
    def _possible_(self,dataset):
        return(dataset.variables.keys())
    
    def _get_dtype_(self,dataset):
        return(dataset.variables[self.name].dtype.str)
    
    def _get_value_(self,dataset):
        ret = dataset.variables[self.name][:]
        ret = self._format_(ret)
        return(ret)
    
    def _format_(self,value):
        return(value)
    
    
class DimensionElement(Element):
    pass


class AttributeElement(Element):
    
    def __init__(self,dataset,parent):
        self._parent = parent
        super(AttributeElement,self).__init__(dataset)
        self.value = self._get_value_(dataset)
    
    def _possible_(self,dataset):
        return(dataset.variables[self._parent.name].ncattrs())
    
    def _get_value_(self,dataset):
        ret = getattr(dataset.variables[self._parent.name],self.name)
        return(ret)
    
    
class DatasetElement(Element):
    
    def __init__(self,dataset):
        super(DatasetElement,self).__init__(dataset)
        self.value = self._get_value_(dataset)
    
    def _possible_(self,dataset):
        return(dataset.ncattrs())
    
    def _get_value_(self,dataset):
        ret = getattr(dataset,self.name)
        return(ret)