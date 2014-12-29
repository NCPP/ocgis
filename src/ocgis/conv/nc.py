import netCDF4 as nc

import datetime
import ocgis
from ocgis.conv.base import AbstractConverter
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
            ret = constants.NETCDF_DEFAULT_DATA_MODEL
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
                    ret = constants.NETCDF_DEFAULT_DATA_MODEL
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

        # append to the history attribute
        history_str = '\n{dt} UTC ocgis-{release}'.format(dt=datetime.datetime.utcnow(), release=ocgis.__release__)
        if self.ops is not None:
            history_str += ': {0}'.format(self.ops)
        original_history_str = ds.__dict__.get('history', '')
        setattr(ds, 'history', original_history_str+history_str)
