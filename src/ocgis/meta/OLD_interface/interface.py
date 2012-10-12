import models
import numpy as np
from ocg.util.helpers import bounding_coords
import itertools
import warnings
import netCDF4 as nc
import datetime


class InterfaceElement(object):
    
    def __init__(self,name,Model,kwd_name=None,var_ref=None):
        self.name = name
        self.Model = Model
        self.kwd_name = kwd_name
        self.var_ref = var_ref
        
    def set(self,target,kwds,dataset):
        if self.kwd_name is not None:
            name = kwds[self.kwd_name]
        elif self.name in kwds:
            name = kwds[self.name]
        else:
            try:
                name = kwds[self.name+'_name']
            except KeyError:
                name = None
        if self.var_ref is None:
            setattr(target,self.name,self.Model(dataset,name=name))
        else:
            setattr(target,self.name,self.Model(getattr(target,self.var_ref),dataset,name=name))
            
            
class NestedInterfaceElement(InterfaceElement):

    def __init__(self,var_ref,name,Model):
        super(NestedInterfaceElement,self).__init__(name,Model,var_ref=var_ref)
        

class Interface(object):
    _contains = []
    _ncontains = []
    
    def __init__(self,dataset,**kwds):
        self.dataset = dataset
        self._process_kwds_(kwds)
        
    def _process_kwds_(self,kwds):
        for c in itertools.chain(self._contains,self._ncontains):
            c.set(self,kwds,self.dataset)


class SpatialInterface(Interface):
    _contains = [
                 InterfaceElement('rowbnds',models.RowBounds),
                 InterfaceElement('colbnds',models.ColumnBounds),
                 InterfaceElement('row',models.Row),
                 InterfaceElement('col',models.Column)
                 ]
    
    def get_bounds(self,colidx):
        col,row = np.meshgrid(self.colbnds.value[:,colidx],
                              self.rowbnds.value[:,colidx])
        ## some data uses 360 dynamic range for longitude coordinates. compliance
        ## with WGS84 data requires data ranging from -180 to 180.
        if col.max() > 180:
#            idx = col > 180
#            col[idx] = -(col[idx] - 180)
            col = col - 180
            warnings.warn(('0 to 360 longitude variable encountered. simple '
                           'remapping to [-180,180] occurred.'))
        return(col,row)
    
    def get_min_bounds(self):
        return(self.get_bounds(0))
    
    def get_max_bounds(self):
        return(self.get_bounds(1))
    
    def subset_bounds(self,polygon):
        bounds = bounding_coords(polygon)
        xbnd = self.colbnds.value[:]
        ybnd = self.rowbnds.value[:]
        xbnd_idx1 = self._subset_(xbnd[:,0],bounds.min_x,bounds.max_x,ret_idx=True,method='open')
        xbnd_idx2 = self._subset_(xbnd[:,1],bounds.min_x,bounds.max_x,ret_idx=True,method='open')
        ybnd_idx1 = self._subset_(ybnd[:,0],bounds.min_y,bounds.max_y,ret_idx=True,method='open')
        ybnd_idx2 = self._subset_(ybnd[:,1],bounds.min_y,bounds.max_y,ret_idx=True,method='open')
        return(xbnd[xbnd_idx1*xbnd_idx2,:],ybnd[ybnd_idx1*ybnd_idx2,:])
    
    def subset_centroids(self,polygon):
        bounds = bounding_coords(polygon)
        y = self._subset_(self.row.value[:],bounds.min_y,bounds.max_y)
        x = self._subset_(self.col.value[:],bounds.min_x,bounds.max_x)
        return(x,y)
    
    def _subset_(self,ary,lower,upper,ret_idx=False,method='open'):
        if method == 'open':
            idx1 = ary >= lower
            idx2 = ary <= upper
        if method == 'closed':
            idx1 = ary > lower
            idx2 = ary < upper
        if not ret_idx:
            return(ary[idx1*idx2])
        else:
            return(idx1*idx2)
    

class TemporalInterface(Interface):
    _contains = [InterfaceElement('time',models.Time)]
    
    _ncontains = [NestedInterfaceElement('time','calendar',models.Calendar),
                  NestedInterfaceElement('time','units',models.TimeUnits)]
    
    def get_timevec(self):
        ret = nc.netcdftime.num2date(self.time.value[:],
                                     self.units.value,
                                     self.calendar.value)
        if not isinstance(ret[0],datetime.datetime):
            reformat_timevec = np.empty(ret.shape,dtype=object)
            for ii,t in enumerate(ret):
                reformat_timevec[ii] = datetime.datetime(t.year,t.month,t.day,
                                                         t.hour,t.minute,t.second)
            ret = reformat_timevec
        return(ret)
    
    
class LevelInterface(Interface):
    _contains = [InterfaceElement('level',models.Level)]