from ocgis.conv.base import OcgConverter
import netCDF4 as nc
from ocgis.interface.projection import WGS84
from ocgis import constants
from ocgis.api.collection import CalcCollection
from ocgis import env

    
class NcConverter(OcgConverter):
    _ext = 'nc'
    
    def _write_(self,file_only=False):
        ## get the collection
        for ii,coll in enumerate(self):
            if ii > 0:
                raise(ValueError('only one collection should be returned for NC conversion'))
        arch = coll._archetype
        ## dataset object to write to
        ds = nc.Dataset(self.path,'w',format=arch.request_dataset.ds._ds.file_format)
        ## reference the interfaces
#        iglobal = arch._i
#        spatial = iglobal.spatial
        grid = arch.spatial.grid
        temporal = arch.temporal
        level = arch.level
        meta = arch.metadata
        
        ## get or make the bounds dimensions
        try:
            bounds_name = list(set(meta['dimensions'].keys()).intersection(set(constants.name_bounds)))[0]
        except IndexError:
            bounds_name = constants.ocgis_bounds
                
        ## add dataset/global attributes
        for key,value in meta['dataset'].iteritems():
            setattr(ds,key,value)
        
#        if isinstance(spatial,SpatialInterfacePoint):
#            is_poly = False
#        else:
#            is_poly = True
#        
#        ## spatial variable calculation ########################################
#        
#        ## first construct values from the stored geometry array
#        geom = arch.spatial._value
#        
#        latitude_values = np.empty(geom.shape[0],dtype=float)
#        longitude_values = np.empty(geom.shape[1],dtype=float)
#        if is_poly:
#            latitude_bounds_values = np.empty((geom.shape[0],2),dtype=float)
#            longitude_bounds_values = np.empty((geom.shape[1],2),dtype=float)
#        ## iterate geometries filling in updated geometry values
#        for row_idx in range(len(geom[:,0])):
#            latitude_values[row_idx] = geom[row_idx,0].centroid.y
#            lon_min,lat_min,lon_max,lat_max = geom[row_idx,0].bounds
#            if is_poly:
#                latitude_bounds_values[row_idx,:] = [lat_min,lat_max]
#        for col_idx in range(len(geom[0,:])):
#            longitude_values[col_idx] = geom[0,col_idx].centroid.x
#            lon_min,lat_min,lon_max,lat_max = geom[0,col_idx].bounds
#            if is_poly:
#                longitude_bounds_values[col_idx,:] = [lon_min,lon_max]

        ## make dimensions #####################################################
        
        ## time dimensions
        dim_temporal = ds.createDimension(temporal.name)
#        
#        if self.mode == 'calc':
#            dim_time = ds.createDimension(temporal.name)
#        else:
#            dim_time = ds.createDimension(temporal.name)

        ## spatial dimensions
        dim_row = ds.createDimension(grid.row.name,grid.row.shape[0])
        dim_col = ds.createDimension(grid.column.name,grid.column.shape[0])
        if arch.spatial.grid.is_bounded:
            dim_bnds = ds.createDimension(bounds_name,2)
        else:
            dim_bnds = None
        
        ## set data + attributes ###############################################
        
        ## time variable
        if temporal.group is not None:
            time_nc_value = temporal.get_nc_time(temporal.group.representative_datetime)
        else:
            time_nc_value = temporal.get_nc_time(arch.temporal.value)
#            time_nc_value = temporal.calculate(arch.temporal.value[:,1])
        ## if bounds are available for the time vector transform those as well
        if temporal.bounds is not None:
            if dim_bnds is None:
                dim_bnds = ds.createDimension(bounds_name,2)
            if temporal.group is not None:
                time_bounds_nc_value = temporal.get_nc_time(temporal.group.bounds)
            else:
                time_bounds_nc_value = temporal.get_nc_time(temporal.bounds)
            if temporal.group is None:
                times_bounds = ds.createVariable(temporal.name_bounds,time_bounds_nc_value.dtype,(dim_temporal._name,bounds_name))
            else:
                times_bounds = ds.createVariable('climatology_'+bounds_name,time_bounds_nc_value.dtype,(dim_temporal._name,bounds_name))
            times_bounds[:] = time_bounds_nc_value
            for key,value in meta['variables'][temporal.name_bounds]['attrs'].iteritems():
                setattr(times_bounds,key,value)
        times = ds.createVariable(temporal.name,time_nc_value.dtype,(dim_temporal._name,))
        times[:] = time_nc_value
        for key,value in meta['variables'][temporal.name]['attrs'].iteritems():
            if temporal.group is not None and key == 'bounds':
                setattr(times,'climatology',times_bounds._name)
            else:
                setattr(times,key,value)
            
        ## level variable
        ## if there is no level on the variable no need to build one.
        if level is None:
            dim_level = None
        ## if there is a level, create the dimension and set the variable.
        else:
            dim_level = ds.createDimension(level.name,len(arch.level.value))
            levels = ds.createVariable(level.name,arch.level.value.dtype,(dim_level._name,))
            levels[:] = arch.level.value
            for key,value in meta['variables'][level.name]['attrs'].iteritems():
                setattr(levels,key,value)
            if level.bounds is not None:
                if dim_bnds is None:
                    dim_bnds = ds.createDimension(bounds_name,2)
                levels_bounds = ds.createVariable(level.name_bounds,arch.level.value.dtype,(dim_level._name,bounds_name))
                levels_bounds[:] = arch.level.bounds
                for key,value in meta['variables'][level.name_bounds]['attrs'].iteritems():
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
#            ret.proj4_str = iglobal.spatial.projection.sr.ExportToProj4()
            ## add variable attributes
            try:
                for key,value in meta['variables'][name]['attrs'].iteritems():
                    setattr(ret,key,value)
            except KeyError:
                pass
            return(ret)
        ## set the spatial data
        _make_spatial_variable_(ds,grid.row.name,grid.row.value,(dim_row,),meta)
        _make_spatial_variable_(ds,grid.column.name,grid.column.value,(dim_col,),meta)
        if grid.is_bounded:
            _make_spatial_variable_(ds,grid.row.name_bounds,grid.row.bounds,(dim_row,dim_bnds),meta)
            _make_spatial_variable_(ds,grid.column.name_bounds,grid.column.bounds,(dim_col,dim_bnds),meta)
        
        ## set the variable(s) #################################################
        
        ## loop through variables
        if isinstance(coll,CalcCollection):
            for calc_name,calc_value in coll.calc[coll.variables.keys()[0]].iteritems():
                value = ds.createVariable(calc_name,calc_value.dtype,
                           value_dims,fill_value=calc_value.fill_value)
                if not file_only:
                    value[:] = calc_value
        else:
            if file_only: raise(NotImplementedError)
            for var_name,var_value in coll.variables.iteritems():
                ## reference level interface
    #            level = var_value.ocg_dataset.i.level
                ## create the value variable.
                value = ds.createVariable(var_name,var_value.value.dtype,
                               value_dims,fill_value=constants.fill_value)
                if not file_only:
                    value[:] = var_value.value
    #            value.fill_value = var_value.raw_value.fill_value
                for key,val in meta['variables'][var_name]['attrs'].iteritems():
                    setattr(value,key,val)
                    
        ## add projection variable if applicable ###############################
        
        if not isinstance(arch.spatial.projection,WGS84):
            arch.spatial.projection.write_to_rootgrp(ds,meta)
        
        ds.close()
