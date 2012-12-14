from collections import OrderedDict


class OcgVariable(object):
    
    def __init__(self,tdim,sdim,ldim=None):
        pass


class OcgIdentifier(OrderedDict):
    
    def __init__(self,*args,**kwds):
        self._curr = 1
        super(OcgIdentifier,self).__init__(*args,**kwds)
    
    def add(self,value):
        if self._get_is_unique_(value):
            self.update({value:self._get_current_identifier_()})
        
    def _get_is_unique_(self,value):
        if value in self:
            ret = False
        else:
            ret = True
        return(ret)
    
    def _get_current_identifier_(self):
        try:
            return(self._curr)
        finally:
            self._curr += 1
