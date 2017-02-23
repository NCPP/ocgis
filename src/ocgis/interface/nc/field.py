from copy import deepcopy

import numpy as np

from ocgis.interface.base.field import Field


class NcField(Field):
    def _get_value_from_source_(self, request_dataset, variable_name):
        # Collect the dimension slices.
        axis_slc = {}
        axis_slc['T'] = self.temporal._src_idx
        try:
            axis_slc['Y'] = self.spatial.grid.row._src_idx
            axis_slc['X'] = self.spatial.grid.col._src_idx
        # If grid and row are not present on the grid, the source indices are attached to the grid object itself.
        except AttributeError:
            axis_slc['Y'] = self.spatial.grid._src_idx['row']
            axis_slc['X'] = self.spatial.grid._src_idx['col']
        if self.realization is not None:
            axis_slc['R'] = self.realization._src_idx
        if self.level is not None:
            # Scalar level dimensions have no place on the data variable.
            if not self.level.is_scalar:
                axis_slc['Z'] = self.level._src_idx

        dim_map = request_dataset.source_metadata['dim_map']

        # Handle scalar level dimensions.
        if self.level is not None and self.level.is_scalar:
            slc = [None for v in dim_map.values() if v is not None and v['dimension'] is not None]
        else:
            slc = [None for v in dim_map.values() if v is not None]

        axes = deepcopy(slc)
        for k, v in dim_map.iteritems():
            if v is not None:
                try:
                    slc[v['pos']] = axis_slc[k]
                except KeyError:
                    # No position for scalar level dimensions.
                    if k == 'Z' and self.level.is_scalar:
                        continue
                    else:
                        raise
                else:
                    axes[v['pos']] = k
        # Ensure axes ordering is as expected.
        possible = [['T', 'Y', 'X'], ['T', 'Z', 'Y', 'X'], ['R', 'T', 'Y', 'X'], ['R', 'T', 'Z', 'Y', 'X']]
        check = [axes == poss for poss in possible]
        try:
            assert any(check)
        except AssertionError:
            # Allow this special case of axis orders. The axes will be swapped following load from source.
            if axes == ['T', 'X', 'Y']:
                swap_row_column = True
            else:
                raise
        else:
            # Leave row and column axes as is.
            swap_row_column = False

        ds = request_dataset.driver.open()
        try:
            try:
                raw = ds.variables[variable_name].__getitem__(slc)
            # If the slc list are all single element numpy vectors, convert to slice objects to avoid index error.
            except IndexError:
                if all([len(a) == 1 for a in slc]):
                    slc2 = [slice(a[0], a[0] + 1) for a in slc]
                    raw = ds.variables[variable_name].__getitem__(slc2)
                else:
                    raise

            # Swap the row/column y/x axes if requested.
            if swap_row_column:
                raw = np.swapaxes(raw, 1, 2)

            # Always return a masked array.
            if not isinstance(raw, np.ma.MaskedArray):
                raw = np.ma.array(raw, mask=False)
            # Reshape the data adding singleton axes where necessary.
            new_shape = [1 if e is None else len(e) for e in [axis_slc.get(a) for a in self._axes]]
            raw = raw.reshape(*new_shape)

            return raw
        finally:
            request_dataset.driver.close(ds)
