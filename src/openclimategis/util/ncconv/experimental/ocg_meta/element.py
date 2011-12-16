import models
import inspect


class PolyElementNotFound(Exception):
    
    def __init__(self,klass):
        self.klass = klass
        
    def __str__(self):
        args = (self.klass.__name__,
                self.klass._names)
        msg = 'PolyElement model "{0}" found no matches for {1}'.format(*args)
        return(msg)


class PolyElement(object):
    """
    _names -- list of possible names. the first element is the desired name in
        the case of writing an output.
    dataset -- open netCDF4-python Dataset object.
    name -- set the name of the element. if None is passed, then it searches
        the dataset for the correct name.
    """
    _names = []
    
    def __init__(self,dataset,name=None):
        self.dataset = dataset
        self.name = name or self.find()
        self._value = None
        
    @property
    def value(self):
        if self._value is None:
            self._value = self.get()
        return(self._value)
        
    def find(self):
        possible = self.possible()
        for ii,poss in enumerate(possible,start=1):
            a = [n == poss for n in self._names]
            if any(a):
                return(self._names[a.index(True)])
            if ii == len(possible):
                raise(PolyElementNotFound(self.__class__))
        
    def get(self):
        raise(NotImplementedError)
    
    def possible(self):
        raise(NotImplementedError)
    
    def set(self,*args,**kwds):
        raise(NotImplementedError)
    
    @staticmethod
    def get_checks():
        from models import Register
        members = inspect.getmembers(models)
        return([mem[1] for mem in members if inspect.isclass(mem[1]) and 
                                             issubclass(mem[1],Register) and 
                                             mem[1] != Register])
    
    
class DatasetPolyElement(PolyElement):
    
    def get(self):
        return(getattr(self.dataset,self.name))
    
    def possible(self):
        return(self.dataset.ncattrs())
    
    
class VariablePolyElement(PolyElement):
    _dtype = None
    
    def __init__(self,*args,**kwds):
        assert(self._dtype is not None)
        super(VariablePolyElement,self).__init__(*args,**kwds)
        
    def get(self):
        return(self.dataset.variables[self.name])
    
    def possible(self):
        return(self.dataset.variables.keys())
    
    def make_dimension_tup(self,*args):
        return([arg.name for arg in args])
    
    
class VariableAttrPolyElement(PolyElement):
    
    def __init__(self,variable_poly_element,*args,**kwds):
        self.variable_poly_element = variable_poly_element
        super(VariableAttrPolyElement,self).__init__(*args,**kwds)
        
    def get(self):
        var = self.dataset.variables[self.variable_poly_element.name]
        return(getattr(var,self.name))
    
    def possible(self):
        return(self.dataset.variables[self.variable_poly_element.name].ncattrs())


class DimensionElement(PolyElement):
    
    def calculate(self,*args,**kwds):
        raise(NotImplementedError)
    
    def get(self):
        return(self.dataset.dimensions[self.name])
    
    def possible(self):
        return(self.dataset.dimensions.keys())
    
    
class SpatialDimensionElement(DimensionElement):
    
    def calculate(self,grid):
        raise(NotImplementedError)
    
    
class TemporalDimensionElement(DimensionElement):
    
    def calculate(self,timevec):
        return(len(timevec))
    
    
class LevelDimensionElement(DimensionElement):
    pass

    
class TranslationalElement(object):
    pass
    
    
class SimpleTranslationalElement(TranslationalElement):
    
    def calculate(self):
        return(NotImplementedError)


class SpatialTranslationalElement(TranslationalElement):
    
    def calculate(self,grid):
        return(NotImplementedError)


class TemporalTranslationalElement(TranslationalElement):
    
    def calculate(self,timevec,**kwds):
        return(NotImplementedError)