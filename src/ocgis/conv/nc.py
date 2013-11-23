from ocgis.conv.base import OcgConverter
import netCDF4 as nc
from ocgis import constants
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.interface.base.crs import CFWGS84
from ocgis.interface.nc.temporal import NcTemporalGroupDimension

    
class NcConverter(OcgConverter):
    _ext = 'nc'
    
    def _finalize_(self,ds):
        ds.close()
        
    def _build_(self,coll):
        ds = nc.Dataset(self.path,'w',format=self._get_file_format_())
        return(ds)
        
    def _get_file_format_(self):
        file_format = set()
        for rd in self.ops.dataset:
            rr = rd._source_metadata['file_format']
            if isinstance(rr,basestring):
                tu = [rr]
            else:
                tu = rr
            file_format.update(tu)
        if len(file_format) > 1:
            exc = ValueError('Multiple file formats found: {0}'.format(file_format))
            ocgis_lh(exc=exc,logger='conv.nc')
        else:
            return(list(file_format)[0])
    
    def _write_coll_(self,ds,coll):
        
        ## get the target field from the collection
        arch = coll._archetype_field
        
        ## reference the interfaces
        grid = arch.spatial.grid
        temporal = arch.temporal
        level = arch.level
        meta = arch.meta
        
        ## get or make the bounds dimensions
        try:
            bounds_name = list(set(meta['dimensions'].keys()).intersection(set(constants.name_bounds)))[0]
        except IndexError:
            bounds_name = constants.ocgis_bounds
                
        ## add dataset/global attributes
        for key,value in meta['dataset'].iteritems():
            setattr(ds,key,value)

        ## make dimensions #####################################################
        
        ## time dimensions
        name_dim_temporal = meta['dim_map']['T']['dimension']
        name_bounds_temporal = meta['dim_map']['T']['bounds']
        name_variable_temporal = meta['dim_map']['T']['variable']
        
        dim_temporal = ds.createDimension(name_dim_temporal)

        ## spatial dimensions
        dim_row = ds.createDimension(grid.row.meta['dimensions'][0],grid.row.shape[0])
        dim_col = ds.createDimension(grid.col.meta['dimensions'][0],grid.col.shape[0])
        if grid.row.bounds is None:
            dim_bnds = None
        else:
            dim_bnds = ds.createDimension(bounds_name,2)
        
        ## set data + attributes ###############################################
        
        ## time variable
        time_nc_value = arch.temporal.value

        ## if bounds are available for the time vector transform those as well
        if isinstance(temporal,NcTemporalGroupDimension):
            if dim_bnds is None:
                dim_bnds = ds.createDimension(bounds_name,2)
            times_bounds = ds.createVariable('climatology_'+bounds_name,time_nc_value.dtype,
                                             (dim_temporal._name,bounds_name))
            times_bounds[:] = temporal.bounds
        elif temporal.bounds is not None:
            if dim_bnds is None:
                dim_bnds = ds.createDimension(bounds_name,2)
            time_bounds_nc_value = temporal.bounds
            times_bounds = ds.createVariable(name_bounds_temporal,time_bounds_nc_value.dtype,(dim_temporal._name,bounds_name))
            times_bounds[:] = time_bounds_nc_value
            for key,value in meta['variables'][name_bounds_temporal]['attrs'].iteritems():
                setattr(times_bounds,key,value)
        times = ds.createVariable(name_variable_temporal,time_nc_value.dtype,(dim_temporal._name,))
        times[:] = time_nc_value
        for key,value in meta['variables'][name_variable_temporal]['attrs'].iteritems():
            setattr(times,key,value)
        
        ## add climatology bounds
        if isinstance(temporal,NcTemporalGroupDimension):
            setattr(times,'climatology','climatology_'+bounds_name)
            
        ## level variable
        ## if there is no level on the variable no need to build one.
        if level is None:
            dim_level = None
        ## if there is a level, create the dimension and set the variable.
        else:
            name_dim_level = meta['dim_map']['Z']['dimension']
            name_bounds_level = meta['dim_map']['Z']['bounds']
            name_variable_level = meta['dim_map']['Z']['variable']
            
            dim_level = ds.createDimension(name_dim_level,len(arch.level.value))
            levels = ds.createVariable(name_variable_level,arch.level.value.dtype,(dim_level._name,))
            levels[:] = arch.level.value
            for key,value in meta['variables'][name_variable_level]['attrs'].iteritems():
                setattr(levels,key,value)
            if level.bounds is not None:
                if dim_bnds is None:
                    dim_bnds = ds.createDimension(bounds_name,2)
                levels_bounds = ds.createVariable(name_bounds_level,arch.level.value.dtype,(dim_level._name,bounds_name))
                levels_bounds[:] = arch.level.bounds
                for key,value in meta['variables'][name_bounds_level]['attrs'].iteritems():
                    setattr(levels,key,value)
        if dim_level is not None:
            value_dims = (dim_temporal._name,dim_level._name,dim_row._name,dim_col._name)
        else:
            value_dims = (dim_temporal._name,dim_row._name,dim_col._name)
            
        ## spatial variables ###################################################
        
        ## create and fill a spatial variable
        def _make_spatial_variable_(ds,name,values,dimension_tuple,meta):
            ret = ds.createVariable(name,values.dtype,[d._name for d in dimension_tuple])
            ret[:] = values
            ## add variable attributes
            try:
                for key,value in meta['variables'][name]['attrs'].iteritems():
                    setattr(ret,key,value)
            except KeyError:
                pass
            return(ret)
        ## set the spatial data        
        _make_spatial_variable_(ds,grid.row.meta['axis']['variable'],grid.row.value,(dim_row,),meta)
        _make_spatial_variable_(ds,grid.col.meta['axis']['variable'],grid.col.value,(dim_col,),meta)
        if grid.row.bounds is not None:
            _make_spatial_variable_(ds,grid.row.meta['axis']['bounds'],grid.row.bounds,(dim_row,dim_bnds),meta)
            _make_spatial_variable_(ds,grid.col.meta['axis']['bounds'],grid.col.bounds,(dim_col,dim_bnds),meta)
        
        ## set the variable(s) #################################################
        
        ## loop through variables
        for variable in arch.variables.itervalues():
            value = ds.createVariable(variable.alias,variable.value.dtype,value_dims,
                                      fill_value=variable.value.fill_value)
            if not self.ops.file_only:
                try:
                    value[:] = variable.value.reshape(*value.shape)
                except:
                    import ipdb;ipdb.set_trace()
            value.setncatts(variable.meta['attrs'])
            ## and the units, converting to string as passing a NoneType will raise
            ## an exception.
            value.units = '' if variable.units is None else variable.units
                    
        ## add projection variable if applicable ###############################
        
        if not isinstance(arch.spatial.crs,CFWGS84):
            arch.spatial.crs.write_to_rootgrp(ds,meta)
