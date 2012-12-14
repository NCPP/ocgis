from collections import OrderedDict
import numpy as np


class OcgVariable(object):
    
    def __init__(self,name,value,temporal,spatial,level=None,uri=None):
        assert(value.shape[0] == len(temporal.value))
        if level is None:
            assert(value.shape[1] == 1)
        assert(np.all(value.shape[2:] == spatial.value.shape))
        
        self.name = name
        self.value = value
        self.temporal = temporal
        self.spatial = spatial
        self.level = level
        self.uri = uri
        
        ## hold aggregated values separate from raw
        self.raw_value = value
        ## hold calculated values
        self.calc_value = OrderedDict()
        
    def group(self,*args,**kwds):
        return(self.temporal.group(*args,**kwds))

   
class OcgCollection(object):
    
    def __init__(self,ugeom=None):
        if ugeom is None:
            ugeom = {'ugid':1,'geom':None}
        self.ugeom = ugeom
        self._tid_name = 'tid'
        self._tbid_name = 'tbid'
        self._tgid_name = 'tgid'
        self._tgbid_name = 'tgbid'
        self._mode = 'raw'
        
        ## variable level identifiers
        self.tid = Identifier()
        self.tbid = BoundsIdentifier()
        self.tgid = None
        self.tgbid = BoundsIdentifier()
        self.lid = Identifier()
        self.lbid = BoundsIdentifier() ## level bounds identifier
        self.gid = Identifier()
        ## collection level identifiers
        self.cid = Identifier() ## calculations
        self.vid = Identifier() ## variables
        self.did = Identifier() ## dataset (uri)
        ## variable storage
        self.variables = {}
        ## calculation storage
        self.calculations = {}
        
    
    def add_calculation(self,var,name,values,tgdim):
        self._mode = 'calc'
        self.cid.add(name)
        
        ## update time group identifiers
        _value = tgdim.value
        _bounds = tgdim.bounds
        _tgbid = self.tgbid
        if self.tgid is None:
            self.tgid = TimeGroupIdentifier(tgdim)
        _tgid = self.tgid
        for idx in range(_value.shape[0]):
            _tgid.add(_value[idx,:])
            _tgbid.add(_bounds[idx,:])
        
        try:
            ref = self.calculations[var.name]
        except KeyError:
            self.calculations.update({var.name:{}})
            ref = self.calculations[var.name]
        ref.update({name:values})


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


class Identifier(OrderedDict):
    
    def __init__(self,*args,**kwds):
        self._curr = 1
        super(Identifier,self).__init__(*args,**kwds)
    
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


class BoundsIdentifier(Identifier):
    
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

class TimeGroupIdentifier(object):
    
    def __init__(self,tgdim):
        self.storage = np.empty((tgdim.value.shape[0],tgdim.value.shape[1]+1),dtype=int)
        self.storage[:,1:] = tgdim.value
        self.storage[:,0] = np.arange(1,self.storage.shape[0]+1)
    
    def add(self,value):
        cmp = (self.storage[:,1:] == value).all(axis=1)
        if not cmp.any():
            shape = self.storage.shape
            self.storage.resize(shape[0]+1,shape[1])
            self.storage[-1,0] = self.storage[-2,0] + 1
            self.storage[-1,1:] = value
            
    def get(self,value):
        cmp = (self.storage[:,1:] == value).all(axis=1)
        return(self.storage[cmp,:])