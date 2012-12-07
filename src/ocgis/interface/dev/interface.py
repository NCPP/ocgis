from warnings import warn


class Interface(object):
    
    def __init__(self,rootgrp,target_var):
        self.target_var = target_var
        self.dim_map = self._get_dimension_map_(rootgrp)
        import ipdb;ipdb.set_trace()
        
    def _get_dimension_map_(self,rootgrp):
        var = rootgrp.variables[self.target_var]
        dims = var.dimensions
        mp = dict.fromkeys(['T','Z','X','Y'])
        
        ## try to pull dimensions
        for dim in dims:
            try:
                dimvar = rootgrp.variables[dim]
                try:
                    axis = getattr(dimvar,'axis')
                except AttributeError:
                    warn('guessing dimension location with "axis" attribute missing')
                    axis = self._guess_by_location_(dims,dim)
                mp[axis] = dimvar
            except KeyError:
                raise(NotImplementedError)
        return(mp)
            
    def _guess_by_location_(self,dims,target):
        mp = {3:{0:'T',1:'Y',2:'X'},
              4:{0:'T',2:'Y',3:'X',1:'Z'}}
        return(mp[len(dims)][dims.index(target)])
        