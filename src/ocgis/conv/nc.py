from ocgis.conv.converter import OcgConverter
import netCDF4 as nc
from ocgis.meta.interface.interface import DummyLevelInterface
import numpy as np
from ocgis.util.helpers import iter_array

    
class NcConverter(OcgConverter):
    _ext = 'nc'

#    def __init__(self,*args,**kwds):
        
        
    def write(self):
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
        geom_masked = coll.geom_masked
        latitude_values = np.ma.array(np.empty(geom_masked.reshape(-1).shape[0],dtype=float),mask=False)
        longitude_values = latitude_values.copy()
        latitude_bounds_values = np.ma.array(np.empty((geom_masked.reshape(-1).shape[0],2),dtype=float),
                                             mask=False)
        longitude_bounds_values = latitude_bounds_values.copy()
        ## iterate geometries filling in updated geometry values
        for idx,geom in iter_array(geom_masked.reshape(-1),return_value=True,use_mask=False):
            lat = geom.centroid.y
            lon = geom.centroid.x
            lon_min,lat_min,lon_max,lat_max = geom.bounds
            mask = geom_masked.mask.reshape(-1)[idx]
            latitude_values[idx] = lat
            latitude_values.mask[idx] = mask
            longitude_values[idx] = lon
            longitude_values.mask[idx] = mask
            latitude_bounds_values[idx][:] = [lat_min,lat_max]
            latitude_bounds_values.mask[idx][:] = [mask,mask]
            longitude_bounds_values[idx][:] = [lon_min,lon_max]
            longitude_bounds_values.mask[idx][:] = [mask,mask]
        
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
            value = ds.createVariable(var_name,var_value.raw_value.dtype,value_dims)
            import ipdb;ipdb.set_trace()
            value[:] = var_value.raw_value
        
        import ipdb;ipdb.set_trace()