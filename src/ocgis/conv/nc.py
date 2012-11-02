from ocgis.conv.converter import OcgConverter
import netCDF4 as nc
from ocgis.meta.interface.interface import DummyLevelInterface
import numpy as np
from ocgis.util.helpers import iter_array

    
class NcConverter(OcgConverter):
    _ext = 'nc'

#    def __init__(self,*args,**kwds):
        
    
    def write(self):
        if self.mode == 'raw':
            ret = self._write_raw_()
        elif self.mode == 'agg':
            raise(NotImplementedError)
        else:
            ret = self._write_calc_()
        return(ret)
    
    def _write_calc_(self):
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
        level = self.ocg_dataset.i.level
        
        ## spatial variable calculation ########################################
        
        ## first construct values from the stored geometry array
        geom = coll.geom
        latitude_values = np.empty(geom.shape[0],dtype=float)
        longitude_values = np.empty(geom.shape[1],dtype=float)
        latitude_bounds_values = np.empty((geom.shape[0],2),dtype=float)
        longitude_bounds_values = np.empty((geom.shape[1],2),dtype=float)
        ## iterate geometries filling in updated geometry values
        for row_idx in range(len(geom[:,0])):
            latitude_values[row_idx] = geom[row_idx,0].centroid.y
            lon_min,lat_min,lon_max,lat_max = geom[row_idx,0].bounds
            latitude_bounds_values[row_idx,:] = [lat_min,lat_max]
        for col_idx in range(len(geom[0,:])):
            longitude_values[col_idx] = geom[0,col_idx].centroid.x
            lon_min,lat_min,lon_max,lat_max = geom[0,col_idx].bounds
            longitude_bounds_values[col_idx,:] = [lon_min,lon_max]

        ## make dimensions #####################################################
        
        ## build the level dimension if one existed in the original nc
        if not isinstance(level,DummyLevelInterface):
            raise(NotImplementedError)
        else:
            dim_level = None
        ## time dimensions
        dim_time = ds.createDimension('d_'+temporal.time.name,len(coll.timevec))
        ## spatial dimensions
        dim_lat = ds.createDimension('d_'+spatial.latitude.name,len(latitude_values))
        dim_lon = ds.createDimension('d_'+spatial.longitude.name,len(longitude_values))
        dim_bnds = ds.createDimension('d_bounds',2)
        
        ## set data + attributes ###############################################
        
        ## level if one exists
        if dim_level is not None:
            raise(NotImplementedError)
        
        ## time variable
        time_nc_value = temporal.time.calculate(coll.timevec)
        times = ds.createVariable(temporal.time.name,time_nc_value.dtype,(dim_time._name,))
        times[:] = time_nc_value
        times.calendar = temporal.time.calendar.value
        times.units = temporal.time.units.value
        
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
        latitude_bounds = _make_spatial_variable_(ds,spatial.latitude_bounds.name,latitude_bounds_values,(dim_lat,dim_bnds))
        longitude_bounds = _make_spatial_variable_(ds,spatial.longitude_bounds.name,longitude_bounds_values,(dim_lon,dim_bnds))
        
        ## set the variable(s)
        if dim_level is not None:
            value_dims = (dim_time._name,dim_level._name,dim_lon._name,dim_lat._name)
        else:
            value_dims = (dim_time._name,dim_lon._name,dim_lat._name)
        for var_name,var_value in coll.variables.iteritems():
            value = ds.createVariable(var_name,var_value.raw_value.dtype,value_dims,fill_value=var_value.raw_value.fill_value)
            value[:] = var_value.raw_value
            value.fill_value = var_value.raw_value.fill_value

        ds.close()        
        return(path)
    
    def _write_raw_(self):
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
        level = self.ocg_dataset.i.level
        
        ## spatial variable calculation ########################################
        
        ## first construct values from the stored geometry array
        geom = coll.geom
        latitude_values = np.empty(geom.shape[0],dtype=float)
        longitude_values = np.empty(geom.shape[1],dtype=float)
        latitude_bounds_values = np.empty((geom.shape[0],2),dtype=float)
        longitude_bounds_values = np.empty((geom.shape[1],2),dtype=float)
        ## iterate geometries filling in updated geometry values
        for row_idx in range(len(geom[:,0])):
            latitude_values[row_idx] = geom[row_idx,0].centroid.y
            lon_min,lat_min,lon_max,lat_max = geom[row_idx,0].bounds
            latitude_bounds_values[row_idx,:] = [lat_min,lat_max]
        for col_idx in range(len(geom[0,:])):
            longitude_values[col_idx] = geom[0,col_idx].centroid.x
            lon_min,lat_min,lon_max,lat_max = geom[0,col_idx].bounds
            longitude_bounds_values[col_idx,:] = [lon_min,lon_max]

        ## make dimensions #####################################################
        
        ## time dimensions
        dim_time = ds.createDimension('d_'+temporal.time.name,len(coll.timevec))
        ## spatial dimensions
        dim_lat = ds.createDimension('d_'+spatial.latitude.name,len(latitude_values))
        dim_lon = ds.createDimension('d_'+spatial.longitude.name,len(longitude_values))
        dim_bnds = ds.createDimension('d_bounds',2)
        
        ## set data + attributes ###############################################
        
        ## time variable
        time_nc_value = temporal.time.calculate(coll.timevec)
        times = ds.createVariable(temporal.time.name,time_nc_value.dtype,(dim_time._name,))
        times[:] = time_nc_value
        times.calendar = temporal.time.calendar.value
        times.units = temporal.time.units.value
        
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
        latitude_bounds = _make_spatial_variable_(ds,spatial.latitude_bounds.name,latitude_bounds_values,(dim_lat,dim_bnds))
        longitude_bounds = _make_spatial_variable_(ds,spatial.longitude_bounds.name,longitude_bounds_values,(dim_lon,dim_bnds))
        
        ## set the variable(s)
        for ii,(var_name,var_value) in enumerate(coll.variables.iteritems()):
            level = var_value.ocg_dataset.i.level
            if isinstance(level,DummyLevelInterface):
                dim_level = None
            else:
                dim_level = ds.createDimension('d_'+level.level.name)
                levels = ds.createVariable(level.level.name,var_value.levelvec.dtype,(dim_level._name,))
                levels[:] = var_value.levelvec
            if dim_level is not None:
                value_dims = (dim_time._name,dim_level._name,dim_lon._name,dim_lat._name)
            else:
                value_dims = (dim_time._name,dim_lon._name,dim_lat._name)
            value = ds.createVariable(var_name,var_value.raw_value.dtype,value_dims,fill_value=var_value.raw_value.fill_value)
            value[:] = var_value.raw_value
            value.fill_value = var_value.raw_value.fill_value

        ds.close()        
        return(path)