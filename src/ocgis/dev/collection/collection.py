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
        self.tid = None
        self.tbid = None
        self.tgid = None
        self.tgbid = None
        self.lid = None
        self.lbid = None ## level bounds identifier
        self.gid = None
        ## collection level identifiers
        self.cid = None ## calculations
        self.vid = None ## variables
        self.did = None ## dataset (uri)
        ## variable storage
        self.variables = {}
        ## calculation storage
        self.calculations = {}
        
    def _uid_(self,attr,init_vals,add=False):
        if getattr(self,attr) is None:
            setattr(self,attr,Identifier(init_vals))
        if add:
            getattr(self,attr).add(init_vals)
            
        return(getattr(self,attr))
    
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
        vid = self._uid_('vid',var.name,add=True)
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


class Identifier(object):
    
    def __init__(self,init_vals):
        if len(init_vals.shape) == 1:
            init_vals = init_vals.reshape(-1,1)
        shape = list(init_vals.shape)
        shape[1] += 1
        self.storage = np.empty(shape,dtype=init_vals.dtype)
        self.storage[:,1:] = init_vals
        self._curr = self.storage.shape[0]+1
        self.storage[:,0] = np.arange(1,self._curr,dtype=init_vals.dtype)
        
    def __len__(self):
        return(self.storage.shape[0])
    
    @property
    def uid(self):
        return(self.storage[:,0].astype(int))
    
    def add(self,value):
        if value.ndim <= 1:
            value = value.reshape(-1,1)
        adds = np.zeros(value.shape[0],dtype=bool)
        for idx in range(adds.shape[0]):
            cmp = self.storage[:,1:] == value[idx,:]
            cmp2 = cmp.all(axis=1)
            if not np.any(cmp2):
                adds[idx] = True
        if adds.any():
            new_values = value[adds,:]
            shape = self.storage.shape
            self.storage.resize(shape[0]+new_values.shape[0],shape[1])
            self.storage[-new_values.shape[0]:,0] = self._get_curr_(new_values.shape[0])
            self.storage[-new_values.shape[0]:,1:] = new_values
            
    def get(self,value):
        cmp = (self.storage[:,1:] == value).all(axis=1)
        return(int(self.storage[cmp,[0]]))
    
    def _get_curr_(self,n):
        ret = np.arange(self._curr,self._curr+n,dtype=self.storage.dtype)
        self._curr = self._curr + n
        return(ret)
