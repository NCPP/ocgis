import logging

from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.util.logging_ocgis import ocgis_lh


class NcVectorDimension(VectorDimension):
    def _get_bounds_from_source_(self):
        # Allow NoneType bounds when there is no request dataset.
        ret = None
        if self._request_dataset is not None:
            assert self.axis is not None

            # Open the connection to the real dataset connection object.
            ds = self._request_dataset.driver.open()
            try:
                # Check for bounds.
                bounds_name = self._request_dataset.source_metadata['dim_map'][self.axis].get('bounds')
                if bounds_name is not None:
                    try:
                        ret = ds.variables[bounds_name][self._src_idx, :]
                    except ValueError:
                        shape = ds.variables[bounds_name]
                        if len(shape) != 2 or shape[1] != 2:
                            msg = (
                                'The bounds variable "{0}" has an improper shape "{1}". Bounds variables should have '
                                'dimensions (m,2).'.format(bounds_name, shape))
                            ocgis_lh(msg=msg, logger='interface.nc', level=logging.WARN)
                        else:
                            raise
            finally:
                self._request_dataset.driver.close(ds)
        return ret

    def _get_value_from_source_(self):
        assert self.axis is not None
        assert self.name is not None

        # Open the connection to the real dataset connection object.
        ds = self._request_dataset.driver.open()
        try:
            try:
                # Reference the variable object.
                var = self._get_variable_from_dataset_(ds, self.name_value)
            except KeyError:
                # For the realization/projection axis, there may in fact be no value associated with it. In it's place,
                # put a standard integer array.
                if self.axis == 'R':
                    var = self._src_idx + 1
                else:
                    raise
            # Get the value.
            ret = var.__getitem__(self._src_idx)
            return ret
        finally:
            self._request_dataset.driver.close(ds)

    def _get_variable_from_dataset_(self, dataset, variable_name):
        return dataset.variables[variable_name]
