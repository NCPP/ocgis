import re
import inspect
import json
from util.ncconv.experimental.helpers import itersubclasses


class OcgFunctionTree(object):
    Groups = None
    
    def __init__(self):
        assert(self.Groups)
        
    def get_function_list(self,functions):
        funcs = []
        for f in functions:
            fname = re.search('([A-Za-z_]+)',f).group(1)
            obj = self.find(fname)()
            try:
                args = re.search('([\d,]+)',f).group(1)
            except AttributeError:
                args = None
            attrs = {'function':getattr(obj,'calculate')}
            if args is not None:
                args = [float(a) for a in args.split(',')]
                attrs.update({'args':args})
            if ':' in f:
                name = f.split(':')[1]
            else:
                name = obj.name
            attrs.update(dict(name=name,desc=obj.description))
            funcs.append(attrs)
        return(funcs)
        
    def json(self):
        children = [Group().format() for Group in self.Groups]
        # ret = {'nodes': dict(expanded=True, children=children)}
        return(json.dumps(children))
    
    def find(self,name):
        for Group in self.Groups:
            for Child in Group().Children:
                if Child().name == name:
                    return(Child)
        raise(AttributeError('function name "{0}" not found'.format(name)))

class OcgFunctionGroup(object):
    name = None
    
    def __init__(self):
        assert(self.name)
        import funcs
        
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


class OcgFunction(object):
    text = None
    description = None
    checked = False
    name = None
    Group = None
    
    def __init__(self):
        if self.text is None:
            self.text = self.__class__.__name__
        if self.name is None:
            self.name = self.text.lower()
        
        assert(self.description is not None)
        assert(self.Group is not None)
    
    @staticmethod
    def calculate(values,**kwds):
        raise(NotImplementedError)
        
    def format(self):
        ret = dict(text=self.text,
                   checked=self.checked,
                   leaf=True,
                   value=self.name,
                   desc=self.description)
        return(ret)
