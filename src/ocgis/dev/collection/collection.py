from collections import OrderedDict
import numpy as np


class OcgVariable(object):
    
    def __init__(self,name,value,temporal,spatial,level=None,uri=None):
        assert(value.shape[0] == len(temporal.value))
        if level is None:
            assert(value.shape[1] == 1)
        assert(np.all(value.shape[2:] == spatial.value.shape))
        
        self.name = name
        self.raw_value = value
        self.temporal = temporal
        self.spatial = spatial
        self.level = level
        self.uri = uri
        
        ## hold aggregated values separate from raw
        self.agg_value = None
        ## hold calculated values
        self.calc_value = OrderedDict()
        
    def group(self,*args,**kwds):
        self.temporal = self.temporal.group(*args,**kwds)
        
        
class OcgCollection(object):
    
    def __init__(self,ugeom=None):
        self.ugeom = ugeom
        self._tid_name = 'tid'
        self._tbid_name = 'tbid'
        
        ## variable level identifiers
        self.tid = OcgIdentifier()
        self.tbid = OcgBoundsIdentifier()
        self.lid = OcgIdentifier()
        self.lbid = OcgBoundsIdentifier() ## level bounds identifier
        self.gid = OcgIdentifier()
        ## collection level identifiers
        self.cid = OcgIdentifier() ## calculations
        self.vid = OcgIdentifier() ## variables
        self.did = OcgIdentifier() ## dataset (uri)
        ## variable storage
        self.variables = {}
        
    def add_variable(self,var):
        ## add the variable to storage
        self.variables.update({var.name:var})
        
        ## update collection identifiers
        self.vid.add(var.name)
        self.did.add(var.uri)
        
        ## update variable identifiers #########################################
        
        ## time & level
        _uid_attr = ['tid','lid']
        _bid_attr = ['tbid','lbid']
        _dim = ['temporal','level']
        _args = (_uid_attr,_bid_attr,_dim)
        if var.level is None:
            for args in _args:
                args.pop()
        for args in zip(*_args):
            args = list(args)
            args.insert(0,var)
            self._add_identifier_(*args)
            
        ## geometry
        masked = var.spatial.get_masked()
        gid = self.gid
        for geom in masked.compressed():
            gid.add(geom.wkb)
    
    def _add_identifier_(self,var,uid_attr,bid_attr,dim):
            tid = getattr(self,uid_attr)
            tbid = getattr(self,bid_attr)
            value = getattr(var,dim).value
            value_bounds = getattr(var,dim).bounds
            
            if value_bounds is None:
                tbid.add(None)
            for idx in range(len(value)):
                tid.add(value[idx])
                if value_bounds is not None:
                    tbid.add(value_bounds[idx])


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


class OcgBoundsIdentifier(OcgIdentifier):
    
    def add(self,value):
        if value is None:
            self[value] = self._get_current_identifier_()
        else:
            try:
                ref = self[value[0]]
            except KeyError:
                self[value[0]] = {}
                ref = self[value[0]]
            try:
                ref[value[1]]
            except KeyError:
                ref[value[1]] = self._get_current_identifier_()

class OcgTimeGroupIdentifier(OcgIdentifier):
    
    def add(self,value):
        import ipdb;ipdb.set_trace()