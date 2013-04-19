from ocgis.conv.converter import OcgConverter
import netCDF4 as nc
from ocgis.interface.interface import SpatialInterfacePoint
import numpy as np
from ocgis.interface.projection import WGS84
from ocgis import constants

    
class NcConverter(OcgConverter):
    _ext = 'nc'
    
    def _write_(self):
        ## get the collection
        for ii,coll in enumerate(self):
            if ii > 0:
                raise(ValueError('only one collection should be returned for NC conversion'))
        ## dataset object to write to
        ds = nc.Dataset(self.path,'w')
        ## reference the interfaces
        arch = coll._arch
        iglobal = arch._i
        spatial = iglobal.spatial
        temporal = iglobal.temporal
        level = iglobal.level
        meta = iglobal._meta
        
        ## get or make the bounds dimensions
        try:
            bounds_name = list(set(meta['dimensions'].keys()).intersection(set(constants.name_bounds)))[0]
        except IndexError:
            bounds_name = constants.ocgis_bounds
                
        ## add dataset/global attributes
        for key,value in meta['dataset'].iteritems():
            setattr(ds,key,value)
        
        if isinstance(spatial,SpatialInterfacePoint):
            is_poly = False
        else:
            is_poly = True
        
        ## spatial variable calculation ########################################
        
        ## first construct values from the stored geometry array
        geom = arch.spatial._value
        
        latitude_values = np.empty(geom.shape[0],dtype=float)
        longitude_values = np.empty(geom.shape[1],dtype=float)
        if is_poly:
            latitude_bounds_values = np.empty((geom.shape[0],2),dtype=float)
            longitude_bounds_values = np.empty((geom.shape[1],2),dtype=float)
        ## iterate geometries filling in updated geometry values
        for row_idx in range(len(geom[:,0])):
            latitude_values[row_idx] = geom[row_idx,0].centroid.y
            lon_min,lat_min,lon_max,lat_max = geom[row_idx,0].bounds
            if is_poly:
                latitude_bounds_values[row_idx,:] = [lat_min,lat_max]
        for col_idx in range(len(geom[0,:])):
            longitude_values[col_idx] = geom[0,col_idx].centroid.x
            lon_min,lat_min,lon_max,lat_max = geom[0,col_idx].bounds
            if is_poly:
                longitude_bounds_values[col_idx,:] = [lon_min,lon_max]

        ## make dimensions #####################################################
        
        ## time dimensions
        if self.mode == 'calc':
            dim_time = ds.createDimension(temporal.name)
        else:
            dim_time = ds.createDimension(temporal.name)
        ## spatial dimensions
        dim_lat = ds.createDimension(spatial.row.name,len(latitude_values))
        dim_lon = ds.createDimension(spatial.col.name,len(longitude_values))
        if is_poly:
            dim_bnds = ds.createDimension(bounds_name,2)
        else:
            dim_bnds = None
        
        ## set data + attributes ###############################################
        
        ## time variable
        if self.mode == 'calc':
            time_nc_value = temporal.calculate(arch.temporal_group.date_centroid)
        else:
            time_nc_value = temporal.calculate(arch.temporal.value[:,1])
        ## if bounds are available for the time vector transform those as well
        if temporal.bounds is not None:
            if dim_bnds is None:
                dim_bnds = ds.createDimension(bounds_name,2)
            if self.mode == 'calc':
                time_bounds_nc_value = temporal.calculate(arch.temporal_group.bounds)
            else:
                time_bounds_nc_value = temporal.calculate(arch.temporal.value[:,(0,2)])
            times_bounds = ds.createVariable(temporal.name_bounds,time_bounds_nc_value.dtype,(dim_time._name,bounds_name))
            times_bounds[:] = time_bounds_nc_value
            for key,value in meta['variables'][temporal.name_bounds]['attrs'].iteritems():
                setattr(times_bounds,key,value)
        times = ds.createVariable(temporal.name,time_nc_value.dtype,(dim_time._name,))
        times[:] = time_nc_value
        for key,value in meta['variables'][temporal.name]['attrs'].iteritems():
            setattr(times,key,value)
            
        ## level variable
        ## if there is no level on the variable no need to build one.
        if level is None:
            dim_level = None
        ## if there is a level, create the dimension and set the variable.
        else:
            dim_level = ds.createDimension(level.name,len(arch.level))
            levels = ds.createVariable(level.name,arch.level.value.dtype,(dim_level._name,))
            levels[:] = arch.level.value[:,1]
            for key,value in meta['variables'][level.name]['attrs'].iteritems():
                setattr(levels,key,value)
            if level.bounds is not None:
                if dim_bnds is None:
                    dim_bnds = ds.createDimension('bounds',2)
                levels_bounds = ds.createVariable(level.name_bounds,arch.level.value.dtype,(dim_level._name,'bounds'))
                levels_bounds[:] = arch.level.value[:,(0,2)]
                for key,value in meta['variables'][level.name_bounds]['attrs'].iteritems():
                    setattr(levels,key,value)
        if dim_level is not None:
            value_dims = (dim_time._name,dim_level._name,dim_lat._name,dim_lon._name)
        else:
            value_dims = (dim_time._name,dim_lat._name,dim_lon._name)
            
        ## spatial variables ###################################################
        
        ## create and fill a spatial variable
        def _make_spatial_variable_(ds,name,values,dimension_tuple,meta):
            ret = ds.createVariable(name,values.dtype,[d._name for d in dimension_tuple])
            ret[:] = values
#            ret.proj4_str = iglobal.spatial.projection.sr.ExportToProj4()
            ## add variable attributes
            for key,value in meta['variables'][name]['attrs'].iteritems():
                setattr(ret,key,value)
            return(ret)
        ## set the spatial data
        latitudes = _make_spatial_variable_(ds,spatial.row.name,latitude_values,(dim_lat,),meta)
        longitudes = _make_spatial_variable_(ds,spatial.col.name,longitude_values,(dim_lon,),meta)
        if is_poly:
            latitude_bounds = _make_spatial_variable_(ds,spatial.row.name_bounds,latitude_bounds_values,(dim_lat,dim_bnds),meta)
            longitude_bounds = _make_spatial_variable_(ds,spatial.col.name_bounds,longitude_bounds_values,(dim_lon,dim_bnds),meta)
        
        ## set the variable(s) #################################################
        
        ## loop through variables
        for var_name,var_value in coll.variables.iteritems():
            if self.mode == 'calc':
                for calc_name,calc_value in var_value.calc_value.iteritems():
                    value = ds.createVariable(calc_name,calc_value.dtype,
                               value_dims,fill_value=var_value.value.fill_value)
                    value[:] = calc_value
            else:
                ## reference level interface
    #            level = var_value.ocg_dataset.i.level
                ## create the value variable.
                value = ds.createVariable(var_name,var_value.value.dtype,
                               value_dims,fill_value=var_value.value.fill_value)
                value[:] = var_value.value
    #            value.fill_value = var_value.raw_value.fill_value
                for key,val in meta['variables'][var_name]['attrs'].iteritems():
                    setattr(value,key,val)
                    
        ## add projection variable if applicable ###############################
        
        if not isinstance(spatial.projection,WGS84):
            spatial.projection.write_to_rootgrp(ds,meta)
        
        ds.close()
