import datetime
import ocgis
from ocgis.conv.base import AbstractConverter
import netCDF4 as nc
from ocgis import constants


class NcConverter(AbstractConverter):
    _ext = 'nc'

    def _finalize_(self, ds):
        ds.close()

    def _build_(self, coll):
        ds = nc.Dataset(self.path, 'w', format=self._get_file_format_())
        return ds

    def _get_file_format_(self):
        file_format = set()
        # if no operations are present, use the default data model
        if self.ops is None:
            ret = constants.netCDF_default_data_model
        else:
            for rd in self.ops.dataset.iter_request_datasets():
                rr = rd.source_metadata['file_format']
                if isinstance(rr, basestring):
                    tu = [rr]
                else:
                    tu = rr
                file_format.update(tu)
            if len(file_format) > 1:
                raise ValueError('Multiple file formats found: {0}'.format(file_format))
            else:
                try:
                    ret = list(file_format)[0]
                except IndexError:
                    # likely all field objects in the dataset. use the default netcdf data model
                    ret = constants.netCDF_default_data_model
        return ret
    
    def _write_coll_(self, ds, coll):
        """
        Write a spatial collection to an open netCDF4 dataset object.

        :param ds: An open dataset object.
        :type ds: :class:`netCDF4.Dataset`
        :param coll: The collection containing data to write.
        :type coll: :class:`~ocgis.SpatialCollection`
        """

        # get the target field from the collection
        arch = coll._archetype_field
        """:type arch: :class:`ocgis.Field`"""

        # get from operations if this is file only.
        try:
            is_file_only = self.ops.file_only
        except AttributeError:
            # no operations object available
            is_file_only = False

        arch.write_to_netcdf_dataset(ds, file_only=is_file_only)

        # ## reference the interfaces
        # grid = arch.spatial.grid
        # temporal = arch.temporal
        # level = arch.level
        # meta = arch.meta
        #
        # # loop through the dimension map, look for a bounds variable, and choose the bounds dimension if possible
        # bounds_name = None
        # for k, v in meta['dim_map'].iteritems():
        #     # it is possible the dimension itself is none
        #     if v is not None and v['bounds'] is not None:
        #         bounds_name = meta['variables'][v['bounds']]['dimensions'][1]
        #         break
        # # if the name of the bounds dimension was not found, choose the default
        # bounds_name = bounds_name or constants.ocgis_bounds
        #
        # ## add dataset/global attributes
        # for key,value in meta['dataset'].iteritems():
        #     setattr(ds,key,value)
        #
        # ## make dimensions #####################################################
        #
        # ## time dimensions
        # name_dim_temporal = meta['dim_map']['T']['dimension']
        # name_bounds_temporal = meta['dim_map']['T']['bounds']
        # name_variable_temporal = meta['dim_map']['T']['variable']
        #
        # dim_temporal = ds.createDimension(name_dim_temporal)
        #
        # ## spatial dimensions
        # dim_row = ds.createDimension(grid.row.meta['dimensions'][0],grid.row.shape[0])
        # dim_col = ds.createDimension(grid.col.meta['dimensions'][0],grid.col.shape[0])
        # if grid.row.bounds is None:
        #     dim_bnds = None
        # else:
        #     dim_bnds = ds.createDimension(bounds_name,2)
        #
        # ## set data + attributes ###############################################
        #
        # ## time variable
        # time_nc_value = arch.temporal.value
        #
        # ## if bounds are available for the time vector transform those as well
        #
        # ## flag to indicate climatology bounds are present and hence the normal
        # ## bounds attribute should be not be added.
        # has_climatology_bounds = False
        #
        # if isinstance(temporal,TemporalGroupDimension):
        #     ## update flag to indicate climatology bounds are present on the
        #     ## output dataset
        #     has_climatology_bounds = True
        #     if dim_bnds is None:
        #         dim_bnds = ds.createDimension(bounds_name,2)
        #     times_bounds = ds.createVariable('climatology_bounds',time_nc_value.dtype,
        #                                      (dim_temporal._name,bounds_name))
        #     times_bounds[:] = temporal.bounds
        #     ## place units and calendar on time dimensions
        #     times_bounds.units = temporal.units
        #     times_bounds.calendar = temporal.calendar
        # elif temporal.bounds is not None:
        #     if dim_bnds is None:
        #         dim_bnds = ds.createDimension(bounds_name,2)
        #     time_bounds_nc_value = temporal.bounds
        #     times_bounds = ds.createVariable(name_bounds_temporal,time_bounds_nc_value.dtype,(dim_temporal._name,bounds_name))
        #     times_bounds[:] = time_bounds_nc_value
        #     for key,value in meta['variables'][name_bounds_temporal]['attrs'].iteritems():
        #         setattr(times_bounds,key,value)
        #     ## place units and calendar on time dimensions
        #     times_bounds.units = temporal.units
        #     times_bounds.calendar = temporal.calendar
        # times = ds.createVariable(name_variable_temporal,time_nc_value.dtype,(dim_temporal._name,))
        # times[:] = time_nc_value
        #
        # ## always place calendar and units on time dimension
        # times.units = temporal.units
        # times.calendar = temporal.calendar
        #
        # ## add time attributes
        # for key,value in meta['variables'][name_variable_temporal]['attrs'].iteritems():
        #     ## leave off the normal bounds attribute
        #     if has_climatology_bounds and key == 'bounds':
        #         if key == 'bounds':
        #             continue
        #     setattr(times,key,value)
        #
        # ## add climatology bounds
        # if isinstance(temporal,TemporalGroupDimension):
        #     setattr(times,'climatology','climatology_bounds')
        #
        # ## level variable
        # ## if there is no level on the variable no need to build one.
        # if level is None:
        #     dim_level = None
        # ## if there is a level, create the dimension and set the variable.
        # else:
        #     name_dim_level = meta['dim_map']['Z']['dimension']
        #     name_bounds_level = meta['dim_map']['Z']['bounds']
        #     name_variable_level = meta['dim_map']['Z']['variable']
        #
        #     dim_level = ds.createDimension(name_dim_level,len(arch.level.value))
        #     levels = ds.createVariable(name_variable_level,arch.level.value.dtype,(dim_level._name,))
        #     levels[:] = arch.level.value
        #     for key,value in meta['variables'][name_variable_level]['attrs'].iteritems():
        #         setattr(levels,key,value)
        #     if level.bounds is not None:
        #         if dim_bnds is None:
        #             dim_bnds = ds.createDimension(bounds_name,2)
        #         levels_bounds = ds.createVariable(name_bounds_level,arch.level.value.dtype,(dim_level._name,bounds_name))
        #         levels_bounds[:] = arch.level.bounds
        #         for key,value in meta['variables'][name_bounds_level]['attrs'].iteritems():
        #             setattr(levels,key,value)
        # if dim_level is not None:
        #     value_dims = (dim_temporal._name,dim_level._name,dim_row._name,dim_col._name)
        # else:
        #     value_dims = (dim_temporal._name,dim_row._name,dim_col._name)
        #
        # ## spatial variables ###################################################
        #
        # ## create and fill a spatial variable
        # def _make_spatial_variable_(ds,name,values,dimension_tuple,meta):
        #     ret = ds.createVariable(name,values.dtype,[d._name for d in dimension_tuple])
        #     ret[:] = values
        #     ## add variable attributes
        #     try:
        #         for key,value in meta['variables'][name]['attrs'].iteritems():
        #             setattr(ret,key,value)
        #     except KeyError:
        #         pass
        #     return(ret)
        # ## set the spatial data
        # _make_spatial_variable_(ds,grid.row.meta['axis']['variable'],grid.row.value,(dim_row,),meta)
        # _make_spatial_variable_(ds,grid.col.meta['axis']['variable'],grid.col.value,(dim_col,),meta)
        # if grid.row.bounds is not None:
        #     _make_spatial_variable_(ds,grid.row.meta['axis']['bounds'],grid.row.bounds,(dim_row,dim_bnds),meta)
        #     _make_spatial_variable_(ds,grid.col.meta['axis']['bounds'],grid.col.bounds,(dim_col,dim_bnds),meta)
        #
        # ## set the variable(s) #################################################
        #
        # ## loop through variables
        # for variable in arch.variables.itervalues():
        #     value = ds.createVariable(variable.alias, variable.dtype, value_dims,
        #                               fill_value=variable.fill_value)
        #     ## if this is a file only operation, set the value, otherwise leave
        #     ## it empty for now.
        #     try:
        #         is_file_only = self.ops.file_only
        #     ## this will happen if there is no operations object.
        #     except AttributeError:
        #         is_file_only = False
        #     if not is_file_only:
        #         value[:] = variable.value.reshape(*value.shape)
        #     value.setncatts(variable.meta['attrs'])
        #     ## and the units, converting to string as passing a NoneType will raise
        #     ## an exception.
        #     value.units = '' if variable.units is None else variable.units
        #
        # ## add projection variable if applicable ###############################
        #
        # if not isinstance(arch.spatial.crs, CFWGS84):
        #     arch.spatial.crs.write_to_rootgrp(ds, meta)

        ## append to the history attribute
        history_str = '\n{dt} UTC ocgis-{release}'.format(dt=datetime.datetime.utcnow(), release=ocgis.__release__)
        if self.ops is not None:
            history_str += ': {0}'.format(self.ops)
        original_history_str = ds.__dict__.get('history', '')
        setattr(ds, 'history', original_history_str+history_str)
