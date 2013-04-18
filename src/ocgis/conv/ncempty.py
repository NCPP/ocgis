from ocgis.conv.converter import OcgConverter
from ocgis.api.dataset.dataset import OcgDataset
import netCDF4 as nc
from ocgis.interface.projection import WGS84
from ocgis.api.dataset.collection.dimension import TemporalDimension
from ocgis import constants


class NcEmpty(OcgConverter):
    _ext = 'nc'
    
    def _set_nc_attrs_(self,var):
        ref = self.meta['variables'][var._name]['attrs']
        for k,v in ref.iteritems():
            setattr(var,k,v)
    
    def _set_variable_(self,key,value,data,set_attrs=True):
        cvar = self.ds.createVariable(key,value['dtype'],
                                      dimensions=value['dimensions'])
        cvar[:] = data
        if set_attrs:
            self._set_nc_attrs_(cvar)
    
    def _write_(self):
        ods = OcgDataset(self.ops.dataset[self.ops.dataset.keys()[0]])
        
        self.ds = nc.Dataset(self.path,'w')
        
        ## reference the interfaces
        gi = ods.i
        spatial = gi.spatial
        temporal = gi.temporal
        level = gi.level
        self.meta = gi._meta
        
        ## add dataset/global attributes
        for key,value in self.meta['dataset'].iteritems():
            setattr(self.ds,key,value)
        
        ## write projection if applicable
        if not isinstance(spatial.projection,WGS84):
            spatial.projection.write_to_rootgrp(self.ds,self.meta)
        
        ## create dimensions except time
        for k,v in self.meta['dimensions'].iteritems():
            if k == temporal.name: continue
            if v['isunlimited']:
                size = None
            else:
                size = v['len']
            self.ds.createDimension(k,size=size)
            
        ## create variables except time
        for k,v in self.meta['variables'].iteritems():
            if level is not None:
                if k == level.name:
                    self._set_variable_(k,v,level.value)
                elif k == level.name_bounds:
                    self._set_variable_(k,v,level.bounds)
            elif k == spatial.row.name:
                self._set_variable_(k,v,spatial.row.value)
            elif k == spatial.row.name_bounds:
                self._set_variable_(k,v,spatial.row.bounds)
            elif k == spatial.col.name:
                self._set_variable_(k,v,spatial.col.value)
            elif k == spatial.col.name_bounds:
                self._set_variable_(k,v,spatial.col.bounds)
        
        ## get or make the bounds dimensions
        inter = set(self.ds.dimensions.keys()).intersection(set(constants.name_bounds))
        if len(inter) == 0:
            dim_bnds = self.ds.createDimension('bounds',size=2)._name
        else:
            dim_bnds = self.ds.dimensions[list(inter)[0]]._name
            
        ## make temporal dimension
        td = TemporalDimension(temporal.tid,temporal.value,bounds=temporal.bounds)
        ## group the data
        if self.ops.calc_grouping is not None:
            tgd = td.group(self.ops.calc_grouping)
            time_value = temporal.calculate(tgd.date_centroid)
            self.ds.createDimension(temporal._dim_name)
            self._set_variable_(temporal.name,
                       {'dtype':time_value.dtype,'dimensions':(temporal._dim_name,)},
                       time_value)
            time_bounds = temporal.calculate(tgd.bounds)
            if temporal.name_bounds is None:
                name_bounds = temporal._dim_name + '_bounds'
                set_attrs = False
            else:
                name_bounds = temporal.name_bounds
                set_attrs = True
            self.ds.variables[temporal.name].bounds = name_bounds
            self._set_variable_(name_bounds,
                       {'dtype':time_bounds.dtype,'dimensions':(temporal._dim_name,dim_bnds)},
                       time_bounds,set_attrs=set_attrs)
        else:
            raise(NotImplementedError)
        
        ## add value variables
        try:
            cdims = (temporal._dim_name,level._dim_name,
                     spatial.row._dim_name,spatial.col._dim_name)
        except AttributeError:
            cdims = (temporal._dim_name,spatial.row._dim_name,spatial.col._dim_name)
        for calc in self.ops.calc:
            self.ds.createVariable(calc['name'],calc['ref'].dtype,dimensions=cdims)
        
        self.ds.close()
        
        return(self.path)
        