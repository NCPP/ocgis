from ocgis.conv.converter import OcgConverter
import netCDF4 as nc
from ocgis.interface.interface import SpatialInterfacePoint
import numpy as np

    
class NcConverter(OcgConverter):
    _ext = 'nc'

    def _write_(self):
        if self.mode == 'raw':
            ret = self._write_raw_()
        elif self.mode == 'agg':
            raise(NotImplementedError)
        else:
            ret = self._write_calc_()
        return(ret)
    
    def _write_calc_(self):
        raise(NotImplementedError)
        ## get the collection
        for ii,(coll,geom_dict) in enumerate(self):
            if ii > 0:
                raise(ValueError('only one collection should be returned for NC conversion'))
        ## output file location
        path = self.get_path()
        ## dataset object to write to
        ds = nc.Dataset(path,'w')
        ## reference the interfaces
        iglobal = self.ocg_dataset.i
        spatial = self.ocg_dataset.i.spatial
        temporal = self.ocg_dataset.i.temporal
        
        if isinstance(spatial,SpatialInterfacePoint):
            is_poly = False
        else:
            is_poly = True
        
        ## spatial variable calculation ########################################
        
        ## first construct values from the stored geometry array
        geom = coll.geom
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
        dim_time = ds.createDimension('d_time_group',len(coll.tgid))
        ## spatial dimensions
        dim_lat = ds.createDimension('d_'+spatial.latitude.name,len(latitude_values))
        dim_lon = ds.createDimension('d_'+spatial.longitude.name,len(longitude_values))
        if is_poly:
            dim_bnds = ds.createDimension('d_bounds',2)
        
        ## set data + attributes ###############################################
        
        ## time variable
        for attr in ['year','month','day']:
            ref = getattr(coll,attr)
            if ref[0] is not None:
                times = ds.createVariable(attr,np.int,(dim_time._name))
                times[:] = ref
#        time_nc_value = temporal.time.calculate(coll.timevec)
#        times = ds.createVariable(temporal.time.name,time_nc_value.dtype,(dim_time._name,))
#        times[:] = time_nc_value
#        times.calendar = temporal.time.calendar.value
#        times.units = temporal.time.units.value
        
        ## spatial variables ###################################################
        
        ## create and fill a spatial variable
        def _make_spatial_variable_(ds,name,values,dimension_tuple):
            ret = ds.createVariable(name,values.dtype,[d._name for d in dimension_tuple])
            ret[:] = values
            ret.projection = iglobal.projection.sr.ExportToProj4()
            return(ret)
        ## set the spatial data
        latitudes = _make_spatial_variable_(ds,spatial.latitude.name,latitude_values,(dim_lat,))
        longitudes = _make_spatial_variable_(ds,spatial.longitude.name,longitude_values,(dim_lon,))
        if is_poly:
            latitude_bounds = _make_spatial_variable_(ds,spatial.latitude_bounds.name,latitude_bounds_values,(dim_lat,dim_bnds))
            longitude_bounds = _make_spatial_variable_(ds,spatial.longitude_bounds.name,longitude_bounds_values,(dim_lon,dim_bnds))
        
        ## set the variable(s) #################################################
        
        ## generator to return all calculations
        def _iter_calc_():
            for var_value in coll.variables.itervalues():
                for calc_name,calc_value in var_value.calc_value.iteritems():
                    yield(var_value,calc_name,calc_value)
            for calc_name,calc_value in coll.calc_multi.iteritems():
                yield(var_value,calc_name,calc_value)
        for var_value,calc_name,calc_value in _iter_calc_():
            ## reference leve interface
            level = var_value.ocg_dataset.i.level
            ## if there is no level on the variable no need to build one.
            if level.is_dummy:
                dim_level = None
            ## if there is a level, create the dimension and set the variable.
            else:
                try:
                    dim_level = ds.createDimension('d_'+level.level.name,len(var_value.levelvec))
                    levels = ds.createVariable(level.level.name,var_value.levelvec.dtype,(dim_level._name,))
                    levels[:] = var_value.levelvec
                except RuntimeError:
                    pass
            if dim_level is not None:
                value_dims = (dim_time._name,dim_level._name,dim_lat._name,dim_lon._name)
            else:
                value_dims = (dim_time._name,dim_lat._name,dim_lon._name)
            ## create the value variable.
            value = ds.createVariable(calc_name,calc_value.dtype,value_dims,fill_value=var_value.raw_value.fill_value)
            value[:] = calc_value
            value.fill_value = var_value.raw_value.fill_value

        ds.close()
        return(path)
    
    def _write_raw_(self):
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
                
        ## add dataset attributes
        for key,value in meta['dataset'].iteritems():
            setattr(ds,key,value)
        
        if isinstance(spatial,SpatialInterfacePoint):
            is_poly = False
        else:
            is_poly = True
        
        ## spatial variable calculation ########################################
        
        ## first construct values from the stored geometry array
        geom = arch.spatial._value
        
#        ##tdk
#        from ocgis.util.helpers import iter_array
#        lc = np.empty(geom.shape,dtype=float)
#        for idx,g in iter_array(geom,return_value=True,use_mask=False):
#            lc[idx] = g.centroid.x
#        import ipdb;ipdb.set_trace()
#        ##tdk
        
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
        dim_time = ds.createDimension(temporal.name,len(arch.temporal))
        ## spatial dimensions
        dim_lat = ds.createDimension(spatial.row.name,len(latitude_values))
        dim_lon = ds.createDimension(spatial.col.name,len(longitude_values))
        if is_poly:
            dim_bnds = ds.createDimension('bounds',2)
        else:
            dim_bnds = None
        
        ## set data + attributes ###############################################
        
        ## time variable
        time_nc_value = temporal.calculate(arch.temporal.value[:,1])
        ## if bounds are available for the time vector transform those as well
        if temporal.bounds is not None:
            if dim_bnds is None:
                dim_bnds = ds.createDimension('bounds',2)
            time_bounds_nc_value = temporal.calculate(arch.temporal.value[:,(0,2)])
            times_bounds = ds.createVariable(temporal.name_bounds,time_bounds_nc_value.dtype,(dim_time._name,'bounds'))
            times_bounds[:] = time_bounds_nc_value
            for key,value in meta['variables'][temporal.name_bounds]['attrs'].iteritems():
                setattr(times_bounds,key,value)
        times = ds.createVariable(temporal.name,time_nc_value.dtype,(dim_time._name,))
        times[:] = time_nc_value
        for key,value in meta['variables'][temporal.name]['attrs'].iteritems():
            setattr(times,key,value)
            
        ## level variable
        ## if there is no level on the variable no need to build one.
        if level.is_dummy:
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
            ret.proj4_str = iglobal.spatial.projection.sr.ExportToProj4()
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
            ## reference level interface
#            level = var_value.ocg_dataset.i.level
            ## create the value variable.
            value = ds.createVariable(var_name,var_value.value.dtype,value_dims,fill_value=var_value.value.fill_value)
            value[:] = var_value.value
#            value.fill_value = var_value.raw_value.fill_value
            for key,val in meta['variables'][var_name]['attrs'].iteritems():
                setattr(value,key,val)
        
        ds.close()
