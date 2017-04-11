import itertools
import logging
import os

import numpy as np
from shapely.geometry import box

from ocgis import Dimension, vm
from ocgis import Variable
from ocgis.base import AbstractOcgisObject
from ocgis.collection.field import OcgField
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.grid import GridXY
from ocgis.test.base import nc_scope
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import VariableCollection
from ocgis.vmachine.mpi import OcgMpi


class GridSplitterConstants(object):
    class IndexFile(object):
        NAME_INDEX_VARIABLE = 'grid_splitter_index'
        NAME_DESTINATION_VARIABLE = 'grid_splitter_destination'
        NAME_Y_BOUNDS_VARIABLE = 'y_bounds'
        NAME_X_BOUNDS_VARIABLE = 'x_bounds'


class GridSplitter(AbstractOcgisObject):
    """
    Splits source and destination grids into separate netCDF files. "Source" is intended to mean the source data for a
    regridding operation. "Destination" is the destination grid for regridding operation.

    The destination subset extents are buffered to ensure full overlap with the source destination subset. Hence,
    elements in the destination subset are globally unique and source subset elements are not necessarily globally
    unique.

    .. note:: All function calls are collective.

    :param src_grid: The source grid for a regridding operation.
    :type src_grid: :class:`ocgis.GridXY`
    :param dst_grid: The destination grid for a regridding operation.
    :type dst_grid: :class:`ocgis.GridXY`
    :param tuple nsplits_dst: The split count for the grid. Tuple length must match the dimension count of the grid.

    >>> npslits_dst = (2, 3)

    :param bool check_contains: If ``True``, check that the source subset bounding box fully contains the destination
     subset bounding box.
    :raises: ValueError
    """

    def __init__(self, src_grid, dst_grid, nsplits_dst, check_contains=True):
        if len(nsplits_dst) != dst_grid.ndim:
            raise ValueError('The number of splits must match the grid dimension count.')

        self.src_grid = src_grid
        self.dst_grid = dst_grid
        self.nsplits_dst = nsplits_dst
        self.check_contains = check_contains

    @staticmethod
    def insert_weighted(index_path, dst_wd, dst_master_path):
        """
        Inserted weighted, destination variable data into the master destination file.

        :param str index_path: Path to the split index netCDF file.
        :param str dst_wd: Working directory containing the destination files holding the weighted data.
        :param str dst_master_path: Path to the destination master file.
        """

        index_field = RequestDataset(index_path).get()
        gs_index_v = index_field[GridSplitterConstants.IndexFile.NAME_INDEX_VARIABLE]
        dst_filenames = gs_index_v.attrs[GridSplitterConstants.IndexFile.NAME_DESTINATION_VARIABLE]
        dst_filenames = index_field[dst_filenames]

        y_bounds = GridSplitterConstants.IndexFile.NAME_Y_BOUNDS_VARIABLE
        y_bounds = gs_index_v.attrs[y_bounds]
        y_bounds = index_field[y_bounds].get_value()

        x_bounds = GridSplitterConstants.IndexFile.NAME_X_BOUNDS_VARIABLE
        x_bounds = gs_index_v.attrs[x_bounds]
        x_bounds = index_field[x_bounds].get_value()

        joined = dst_filenames.join_string_value()
        dst_master_field = RequestDataset(dst_master_path).get()
        for data_variable in dst_master_field.data_variables:
            assert data_variable.ndim == 3
            assert not data_variable.has_allocated_value
            for time_index in range(dst_master_field.time.shape[0]):
                for vidx, source_path in enumerate(joined):
                    source_path = os.path.join(dst_wd, source_path)
                    slc = {dst_master_field.time.dimensions[0].name: time_index,
                           dst_master_field.y.dimensions[0].name: slice(None),
                           dst_master_field.x.dimensions[0].name: slice(None)}
                    source_data = RequestDataset(source_path).get()[data_variable.name][slc]
                    assert not source_data.has_allocated_value
                    with nc_scope(dst_master_path, 'a') as ds:
                        ds.variables[data_variable.name][time_index, y_bounds[vidx][0]:y_bounds[vidx][1],
                        x_bounds[vidx][0]:x_bounds[vidx][1]] = source_data.get_value()

    def iter_dst_grid_slices(self):
        """
        Yield global slices for the destination grid using guidance from ``nsplits_dst``.

        :return: A dictionary with keys as the grid dimension names and the values the associated slice for that
         dimension.
        :rtype: dict

        >>> example_yield = {'dimx': slice(2, 4), 'dimy': slice(10, 20)}
        """

        slice_store = []
        ydim_name = self.dst_grid.dimensions[0].name
        xdim_name = self.dst_grid.dimensions[1].name
        dst_grid_shape_global = self.dst_grid.shape_global
        for idx in range(self.dst_grid.ndim):
            splits = self.nsplits_dst[idx]
            size = dst_grid_shape_global[idx]
            slices = create_slices_for_dimension(size, splits)
            slice_store.append(slices)
        for slice_y, slice_x in itertools.product(*slice_store):
            yield {ydim_name: create_slice_from_tuple(slice_y),
                   xdim_name: create_slice_from_tuple(slice_x)}

    def iter_dst_grid_subsets(self, yield_slice=False):
        """
        Using slices from ``iter_dst_grid_slices``, yield destination grid subsets.

        :param bool yield_slice: If ``True``, yield the slice used on the destination grid.
        :return: The sliced grid object.
        :rtype: :class:`ocgis.GridXY`
        """

        for slc in self.iter_dst_grid_slices():
            sub = self.dst_grid.get_distributed_slice(slc)
            if yield_slice:
                yield sub, slc
            else:
                yield sub

    def iter_src_grid_subsets(self, yield_dst=False):
        """
        Yield source grid subsets using the extent of its associated destination grid subset.

        :param bool yield_dst: If ``True``, yield the destination subset as well as the source grid subset.
        :return: The source grid if ``yield_dst`` is ``False``, otherwise a three-element tuple in the form
         ``(<source grid subset>, <destination grid subset>, <destination grid slice>)``.
        :rtype: :class:`ocgis.GridXY` or (:class:`ocgis.GridXY`, :class:`ocgis.GridXY`, dict)
        """

        if yield_dst:
            yield_slice = True
        else:
            yield_slice = False

        dst_grid_resolution = self.dst_grid.resolution
        src_grid_resolution = self.src_grid.resolution

        if dst_grid_resolution <= src_grid_resolution:
            target_resolution = dst_grid_resolution
        else:
            target_resolution = src_grid_resolution
        buffer_value = 2 * target_resolution

        for yld in self.iter_dst_grid_subsets(yield_slice=yield_slice):
            if yield_slice:
                dst_grid_subset, dst_slice = yld
            else:
                dst_grid_subset = yld

            dst_box = None
            with vm.scoped_by_emptyable('extent_global', dst_grid_subset):
                if not vm.is_null:
                    if self.check_contains:
                        dst_box = box(*dst_grid_subset.extent_global)

                    gmask = self.src_grid.get_mask()
                    assert gmask is None

                    gmask = dst_grid_subset.get_mask()
                    assert gmask is None

                    # Use the envelope! A buffer returns "fancy" borders. We just want to expand the bounding box.
                    sub_box = box(*dst_grid_subset.extent_global).buffer(buffer_value).envelope

                    ocgis_lh(msg=str(sub_box.bounds), level=logging.DEBUG)

                    gmask = self.src_grid.get_mask()
                    assert gmask is None
                else:
                    sub_box, dst_box = [None, None]

            live_ranks = vm.get_live_ranks_from_object(dst_grid_subset)
            sub_box = vm.bcast(sub_box, root=live_ranks[0])

            if self.check_contains:
                dst_box = vm.bcast(dst_box, root=live_ranks[0])

            src_grid_subset = self.src_grid.get_intersects(sub_box, keep_touches=False)

            gmask = self.src_grid.get_mask()
            assert gmask is None

            with vm.scoped_by_emptyable('src_grid_subset', src_grid_subset):
                if not vm.is_null:
                    gmask = src_grid_subset.get_mask()
                    assert gmask is None or not gmask.any()

                    if self.check_contains:
                        src_box = box(*src_grid_subset.extent_global)
                        if not does_contain(src_box, dst_box):
                            raise ValueError('Contains check failed.')
                else:
                    src_grid_subset = GridXY(Variable('x', is_empty=True), Variable('y', is_empty=True))

            if yield_dst:
                yld = (src_grid_subset, dst_grid_subset, dst_slice)
            else:
                yld = src_grid_subset
            yield yld

    def write_subsets(self, src_template, dst_template, wgt_template, index_path):
        """
        Write grid subsets to netCDF files using the provided filename templates. The template must contain the full
        file path with a single curly-bracer pair to insert the combination counter. ``wgt_template`` should not be a
        full path. This name is used when generating weight files.

        >>> template_example = '/path/to/data_{}.nc'

        :param str src_template: The template for the source subset file.
        :param str dst_template: The template for the destination subset file.
        :param str wgt_template: The template for the weight filename.

        >>> wgt_template = 'esmf_weights_{}.nc'

        :param index_path: Path to the output indexing netCDF.
        """

        src_filenames = []
        dst_filenames = []
        wgt_filenames = []
        dst_slices = []

        # nzeros = len(str(reduce(lambda x, y: x * y, self.nsplits_dst)))

        for ctr, (sub_src, sub_dst, dst_slc) in enumerate(self.iter_src_grid_subsets(yield_dst=True), start=1):
            # padded = create_zero_padded_integer(ctr, nzeros)

            src_path = src_template.format(ctr)
            dst_path = dst_template.format(ctr)
            wgt_filename = wgt_template.format(ctr)

            src_filenames.append(os.path.split(src_path)[1])
            dst_filenames.append(os.path.split(dst_path)[1])
            wgt_filenames.append(wgt_filename)
            dst_slices.append(dst_slc)

            for target, path in zip([sub_src, sub_dst], [src_path, dst_path]):
                if target.is_empty:
                    is_empty = True
                    target = None
                else:
                    is_empty = False
                field = OcgField(grid=target, is_empty=is_empty)
                ocgis_lh(msg='writing: {}'.format(path), level=logging.DEBUG)
                with vm.scoped_by_emptyable('field.write', field):
                    if not vm.is_null:
                        field.write(path)
                ocgis_lh(msg='finished writing: {}'.format(path), level=logging.DEBUG)

        with vm.scoped('index write', [0]):
            if not vm.is_null:
                dim = Dimension('nfiles', len(src_filenames))
                vname = ['source_filename', 'destination_filename', 'weights_filename']
                values = [src_filenames, dst_filenames, wgt_filenames]
                grid_splitter_destination = GridSplitterConstants.IndexFile.NAME_DESTINATION_VARIABLE
                attrs = [{'esmf_role': 'grid_splitter_source'},
                         {'esmf_role': grid_splitter_destination},
                         {'esmf_role': 'grid_splitter_weights'}]

                vc = VariableCollection()

                grid_splitter_index = GridSplitterConstants.IndexFile.NAME_INDEX_VARIABLE
                vidx = Variable(name=grid_splitter_index)
                vidx.attrs['esmf_role'] = grid_splitter_index
                vidx.attrs['grid_splitter_source'] = 'source_filename'
                vidx.attrs[GridSplitterConstants.IndexFile.NAME_DESTINATION_VARIABLE] = 'destination_filename'
                vidx.attrs['grid_splitter_weights'] = 'weights_filename'
                x_bounds = GridSplitterConstants.IndexFile.NAME_X_BOUNDS_VARIABLE
                vidx.attrs[x_bounds] = x_bounds
                y_bounds = GridSplitterConstants.IndexFile.NAME_Y_BOUNDS_VARIABLE
                vidx.attrs[y_bounds] = y_bounds
                vc.add_variable(vidx)

                for idx in range(len(vname)):
                    v = Variable(name=vname[idx], dimensions=dim, dtype=str, value=values[idx], attrs=attrs[idx])
                    vc.add_variable(v)

                bounds_dimension = Dimension(name='bounds', size=2)
                xb = Variable(name=x_bounds, dimensions=[dim, bounds_dimension], attrs={'esmf_role': 'x_split_bounds'},
                              dtype=int)
                yb = Variable(name=y_bounds, dimensions=[dim, bounds_dimension], attrs={'esmf_role': 'y_split_bounds'},
                              dtype=int)

                x_name = self.dst_grid.x.dimensions[0].name
                y_name = self.dst_grid.y.dimensions[0].name
                for idx, slc in enumerate(dst_slices):
                    xb.get_value()[idx, :] = slc[x_name].start, slc[x_name].stop
                    yb.get_value()[idx, :] = slc[y_name].start, slc[y_name].stop
                vc.add_variable(xb)
                vc.add_variable(yb)

                vc.write(index_path)

        vm.barrier()


def create_slice_from_tuple(tup):
    return slice(tup[0], tup[1])


def create_slices_for_dimension(size, splits):
    ompi = OcgMpi(size=splits)
    dimname = 'foo'
    ompi.create_dimension(dimname, size, dist=True)
    ompi.update_dimension_bounds()
    slices = []
    for rank in range(splits):
        dimension = ompi.get_dimension(dimname, rank=rank)
        slices.append(dimension.bounds_local)
    return slices


def does_contain(container, containee):
    intersection = container.intersection(containee)
    return np.isclose(intersection.area, containee.area)
