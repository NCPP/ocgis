from ocgis.util.helpers import itersubclasses
import abc


class OcgFunctionGroup(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def name(self): str
    
    def __init__(self):
        from base import OcgFunction
        
        assert(self.name)
        
        import library
        
        self.Children = []
        for sc in itersubclasses(OcgFunction):
            if sc.Group == self.__class__:
                self.Children.append(sc)
        
    def format(self):
        children = [Child().format() for Child in self.Children]
        ret = dict(text=self.name,
                   expanded=True,
                   children=children)
        return(ret)

class BasicStatistics(OcgFunctionGroup):
    name = 'Basic Statistics'
    
    
class Thresholds(OcgFunctionGroup):
    name = 'Thresholds'
    

class MultivariateStatistics(OcgFunctionGroup):
    name = 'Multivariate Statistics'
    
    
class Percentiles(OcgFunctionGroup):
    name = 'Percentiles'