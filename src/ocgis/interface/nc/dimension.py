from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.util.helpers import get_reduced_slice
import logging


class NcVectorDimension(VectorDimension):
    
    def _set_value_from_source_(self):
        ## open the connection to the real dataset connection object
        ds = self._data.driver.open()
        try:
            ## get the variable
            try:
                var = ds.variables[self.meta['name']]
            except KeyError as e:
                ## for the realization/projection axis, there may in fact be no
                ## value associated with it. in it's place, put a standard integer
                ## array.
                if self.axis == 'R':
                    var = self._src_idx + 1
                else:
                    ocgis_lh(logger='interface.nc',exc=e)
            # format the slice
#            slc = get_reduced_slice(self._src_idx)
            ## set the value
            self._value = var.__getitem__(self._src_idx)
            ## now, we should check for bounds here as the inheritance for making
            ## this process more transparent is not in place.
            bounds_name = self._data.source_metadata['dim_map'][self.axis].get('bounds')
            if bounds_name is not None:
                try:
                    self.bounds = ds.variables[bounds_name][self._src_idx,:]
                except ValueError as e:
                    shape = ds.variables[bounds_name]
                    if len(shape) != 2 or shape[1] != 2:
                        msg = 'The bounds variable "{0}" has an improper shape "{1}". Bounds variables should have dimensions (m,2).'.format(bounds_name,shape)
                        ocgis_lh(msg=msg,logger='interface.nc',level=logging.WARN)
                    else:
                        ocgis_lh(exc=e,logger='interface.nc')
        finally:
            self._data.driver.close(ds)
