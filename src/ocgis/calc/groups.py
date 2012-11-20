from ocgis.util.helpers import itersubclasses


class OcgFunctionGroup(object):
    name = None
    
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