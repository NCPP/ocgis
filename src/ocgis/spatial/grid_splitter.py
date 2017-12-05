import itertools
import logging
import os

import netCDF4 as nc
import numpy as np
from shapely.geometry import box

from ocgis import Dimension, vm
from ocgis import Variable
from ocgis.base import AbstractOcgisObject
from ocgis.collection.field import Field
from ocgis.constants import GridSplitterConstants, RegriddingRole, Topology
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.grid import GridUnstruct
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import VariableCollection
from ocgis.variable.geom import GeometryVariable
from ocgis.vmachine.mpi import OcgDist, redistribute_by_src_idx


class GridSplitter(AbstractOcgisObject):
    """
    Splits source and destination grids into separate netCDF files. "Source" is intended to mean the source data for a
    regridding operation. "Destination" is the destination grid for regridding operation.

    The destination subset extents are buffered to ensure full overlap with the source destination subset. Hence,
    elements in the destination subset are globally unique and source subset elements are not necessarily globally
    unique.

    .. note:: Grid parent variable collections may be altered during initializations to account for global source
     indexing.

    .. note:: All function calls are collective.

    :param src_grid: The source grid for a regridding operation.
    :type src_grid: :class:`~ocgis.spatial.grid.AbstractGrid`
    :param dst_grid: The destination grid for a regridding operation.
    :type dst_grid: :class:`~ocgis.spatial.grid.AbstractGrid`
    :param tuple nsplits_dst: The split count for the grid. Tuple length must match the dimension count of the grid.

    >>> npslits_dst = (2, 3)

    :param dict paths: Dictionary of paths used by the grid splitter. Defaults are provided.

    ============ ======================== ===============================================================================================
    Key (str)    Default                  Description
    ============ ======================== ===============================================================================================
    wd           ``os.getcwd()``          Working directory to write to or containing split files, weight files, and splitter index file.
    dst_template ``'split_dst_{}.nc'``    Destination filename template.
    src_template ``'split_src_{}.nc'``    Source filename template.
    wgt_template ``'esmf_weights_{}.nc'`` Weight filename template.
    index_file   ``'01-split_index.nc'``  Name of the index file.
    ============ ======================== ===============================================================================================

    :param bool check_contains: If ``True``, check that the source subset bounding box fully contains the destination
     subset bounding box. Works when coordinate data is ordered and packed similarly between source and destination.
    :param bool allow_masked: If ``True``, allow masked values following a subset.
    :param float src_grid_resolution: Overload the source grid resolution. This is useful when using unstructured data
     that may have a regular patterning to leverage for performance.
    :param float dst_grid_resolution: Same as ``src_grid_resolution`` exception for the destination grid.
    :param bool optimized_bbox_subset: If ``True``, use optimizations for subsetting. If ``False``, do not use
     optimizations. Optimizations are generally okay for structured, rectilinear grids. Optimizations will avoid
     constructing geometries for the subset target. Hence, subset operations with complex boundary definitions should
     generally avoid optimizations (set to ``False``). If ``'auto'``, attempt to identify the best optimization method.
    :param iter_dst: A generator yielding destination grids. This generator must also write the grid.
    :type iter_dst: <generator function>
    :param float buffer_value: The value in units of the destination grid, used to buffer the spatial extent for
     subsetting the source grid. It is best to keep this small, but it must ensure the destination subset is fully
     mapped by the source for whatever purpose the grid splitter is used. If ``None``, the default is double the highest
     resolution between source and destination grids.
    :param bool redistribute: If ``True``, redistribute the source subset for unstructured grids. The redistribution
     reloads the data from source so should not be used with in-memory grids.
    :raises: ValueError
    """

    def __init__(self, src_grid, dst_grid, nsplits_dst, paths=None, check_contains=False, allow_masked=True,
                 src_grid_resolution=None, dst_grid_resolution=None, optimized_bbox_subset='auto', iter_dst=None,
                 buffer_value=None, redistribute=False):
        # TODO: Need to test with an unstructured grid as destination.

        if len(nsplits_dst) != dst_grid.ndim:
            raise ValueError('The number of splits must match the grid dimension count.')

        self.src_grid = src_grid
        self.dst_grid = dst_grid
        self.nsplits_dst = nsplits_dst
        self.check_contains = check_contains
        self.allow_masked = allow_masked
        self.src_grid_resolution = src_grid_resolution
        self.dst_grid_resolution = dst_grid_resolution
        self.iter_dst = iter_dst
        self.buffer_value = buffer_value
        self.redistribute = redistribute

        # Call each grid's grid splitter initialize routine.
        self.src_grid._gs_initialize_(RegriddingRole.SOURCE)
        self.dst_grid._gs_initialize_(RegriddingRole.DESTINATION)

        # Check optimizations.
        assert optimized_bbox_subset is not None
        if optimized_bbox_subset == 'auto':
            # It is okay to use optimization if resolution are provided for unstructured grids. This implies some
            # regular structure for the elements.
            if (is_unstructured(self.src_grid) and src_grid_resolution is None) or \
                    (is_unstructured(self.dst_grid) and dst_grid_resolution is None):
                optimized_bbox_subset = False
            else:
                optimized_bbox_subset = True
        self.optimized_bbox_subset = optimized_bbox_subset

        # Construct default paths if None are provided.
        defaults = {'dst_template': 'split_dst_{}.nc',
                    'src_template': 'split_src_{}.nc',
                    'wgt_template': 'esmf_weights_{}.nc',
                    'index_file': '01-split_index.nc'}
        if paths is None:
            paths = defaults
        else:
            for k, v in defaults.items():
                if k not in paths:
                    paths[k] = v
        if 'wd' not in paths:
            paths['wd'] = os.getcwd()
        self.paths = paths

    def create_full_path_from_template(self, key, index=None):
        ret = self.paths[key]
        ret = os.path.join(self.paths['wd'], ret)
        if index is not None:
            ret = ret.format(index)
        return ret

    def create_merged_weight_file(self, merged_weight_filename, strict=False):
        """
        Merge weight file chunks to a single, global weight file.

        :param str merged_weight_filename: Path to the merged weight file.
        :param bool strict: If ``False``, allow "missing" files where the iterator index cannot create a found file.
         It is best to leave these ``False`` as not all source and destinations are mapped. If ``True``, raise an
        """

        if vm.size > 1:
            raise ValueError("'create_merged_weight_file' does not work in parallel")

        index_filename = self.create_full_path_from_template('index_file')
        ifile = RequestDataset(uri=index_filename).get()
        ifile.load()
        ifc = GridSplitterConstants.IndexFile
        gidx = ifile[ifc.NAME_INDEX_VARIABLE].attrs

        src_global_shape = gidx[ifc.NAME_SRC_GRID_SHAPE]
        dst_global_shape = gidx[ifc.NAME_DST_GRID_SHAPE]

        # Get the global weight dimension size.
        n_s_size = 0
        weight_filename = ifile[gidx[ifc.NAME_WEIGHTS_VARIABLE]]
        wv = weight_filename.join_string_value()
        split_weight_file_directory = self.paths['wd']
        for wfn in map(lambda x: os.path.join(split_weight_file_directory, x), wv):
            if not os.path.exists(wfn):
                if strict:
                    raise IOError(wfn)
                else:
                    continue
            n_s_size += RequestDataset(wfn).get().dimensions['n_s'].size

        # Create output weight file.
        wf_varnames = ['row', 'col', 'S']
        wf_dtypes = [np.int32, np.int32, np.float64]
        vc = VariableCollection()
        dim = Dimension('n_s', n_s_size)
        for w, wd in zip(wf_varnames, wf_dtypes):
            var = Variable(name=w, dimensions=dim, dtype=wd)
            vc.add_variable(var)
        vc.write(merged_weight_filename)

        # Transfer weights to the merged file.
        sidx = 0
        src_indices = self.src_grid._gs_create_global_indices_(src_global_shape)
        dst_indices = self.dst_grid._gs_create_global_indices_(dst_global_shape)

        out_wds = nc.Dataset(merged_weight_filename, 'a')
        for ii, wfn in enumerate(map(lambda x: os.path.join(split_weight_file_directory, x), wv)):
            if not os.path.exists(wfn):
                if strict:
                    raise IOError(wfn)
                else:
                    continue
            wdata = RequestDataset(wfn).get()
            for wvn in wf_varnames:
                odata = wdata[wvn].get_value()
                try:
                    split_grids_directory = self.paths['wd']
                    odata = self._gs_remap_weight_variable_(ii, wvn, odata, src_indices, dst_indices, ifile, gidx,
                                                            split_grids_directory=split_grids_directory)
                except IndexError as e:
                    msg = "Weight filename: '{}'; Weight Variable Name: '{}'. {}".format(wfn, wvn, e.message)
                    raise IndexError(msg)
                out_wds[wvn][sidx:sidx + odata.size] = odata
                out_wds.sync()
            sidx += odata.size
        out_wds.close()

    @staticmethod
    def insert_weighted(index_path, dst_wd, dst_master_path):
        """
        Inserted weighted, destination variable data into the master destination file.

        :param str index_path: Path to the split index netCDF file.
        :param str dst_wd: Working directory containing the destination files holding the weighted data.
        :param str dst_master_path: Path to the destination master weight file.
        """

        index_field = RequestDataset(index_path).get()
        gs_index_v = index_field[GridSplitterConstants.IndexFile.NAME_INDEX_VARIABLE]
        dst_filenames = gs_index_v.attrs[GridSplitterConstants.IndexFile.NAME_DESTINATION_VARIABLE]
        dst_filenames = index_field[dst_filenames]

        y_bounds = GridSplitterConstants.IndexFile.NAME_Y_DST_BOUNDS_VARIABLE
        y_bounds = gs_index_v.attrs[y_bounds]
        y_bounds = index_field[y_bounds].get_value()

        x_bounds = GridSplitterConstants.IndexFile.NAME_X_DST_BOUNDS_VARIABLE
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
                    with nc.Dataset(dst_master_path, 'a') as ds:
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
        :rtype: :class:`ocgis.Grid`
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
        :rtype: :class:`ocgis.Grid` or (:class:`ocgis.Grid`, :class:`ocgis.Grid`, dict)
        """

        if yield_dst:
            yield_slice = True
        else:
            yield_slice = False

        if self.buffer_value is None:
            try:
                if self.dst_grid_resolution is None:
                    dst_grid_resolution = self.dst_grid.resolution
                else:
                    dst_grid_resolution = self.dst_grid_resolution
                if self.src_grid_resolution is None:
                    src_grid_resolution = self.src_grid.resolution
                else:
                    src_grid_resolution = self.src_grid_resolution

                if dst_grid_resolution <= src_grid_resolution:
                    target_resolution = dst_grid_resolution
                else:
                    target_resolution = src_grid_resolution
                buffer_value = 2. * target_resolution
            except NotImplementedError:
                # Unstructured grids do not have an associated resolution.
                if isinstance(self.src_grid, GridUnstruct) or isinstance(self.dst_grid, GridUnstruct):
                    buffer_value = None
                else:
                    raise
        else:
            buffer_value = self.buffer_value

        dst_grid_wrapped_state = self.dst_grid.wrapped_state
        dst_grid_crs = self.dst_grid.crs

        # Use a destination grid iterator if provided.
        if self.iter_dst is not None:
            iter_dst = self.iter_dst(self, yield_slice=yield_slice)
        else:
            iter_dst = self.iter_dst_grid_subsets(yield_slice=yield_slice)

        # Loop over each destination grid subset.
        for yld in iter_dst:
            if yield_slice:
                dst_grid_subset, dst_slice = yld
            else:
                dst_grid_subset = yld

            dst_box = None
            with vm.scoped_by_emptyable('extent_global', dst_grid_subset):
                if not vm.is_null:
                    if self.check_contains:
                        dst_box = box(*dst_grid_subset.extent_global)

                    # Use the envelope! A buffer returns "fancy" borders. We just want to expand the bounding box.
                    extent_global = dst_grid_subset.parent.attrs.get('extent_global')
                    if extent_global is None:
                        extent_global = dst_grid_subset.extent_global
                    sub_box = box(*extent_global)
                    if buffer_value is not None:
                        sub_box = sub_box.buffer(buffer_value).envelope

                    ocgis_lh(msg=str(sub_box.bounds), level=logging.DEBUG)
                else:
                    sub_box, dst_box = [None, None]

            live_ranks = vm.get_live_ranks_from_object(dst_grid_subset)
            sub_box = vm.bcast(sub_box, root=live_ranks[0])

            if self.check_contains:
                dst_box = vm.bcast(dst_box, root=live_ranks[0])

            sub_box = GeometryVariable.from_shapely(sub_box, is_bbox=True, wrapped_state=dst_grid_wrapped_state,
                                                    crs=dst_grid_crs)
            src_grid_subset, src_grid_slice = self.src_grid.get_intersects(sub_box, keep_touches=False, cascade=False,
                                                                           optimized_bbox_subset=self.optimized_bbox_subset,
                                                                           return_slice=True)

            # Reload the data using a new source index distribution.
            if hasattr(src_grid_subset, 'reduce_global'):
                # Only redistribute if we have one live rank.
                if self.redistribute and len(vm.get_live_ranks_from_object(src_grid_subset)) > 0:
                    topology = src_grid_subset.abstractions_available[Topology.POLYGON]
                    cindex = topology.cindex
                    redist_dimname = self.src_grid.abstractions_available[Topology.POLYGON].element_dim.name
                    if src_grid_subset.is_empty:
                        redist_dim = None
                    else:
                        redist_dim = topology.element_dim
                    redistribute_by_src_idx(cindex, redist_dimname, redist_dim)

            with vm.scoped_by_emptyable('src_grid_subset', src_grid_subset):
                if not vm.is_null:
                    if not self.allow_masked:
                        gmask = src_grid_subset.get_mask()
                        if gmask is not None and gmask.any():
                            raise ValueError('Masked values in source grid subset.')

                    if self.check_contains:
                        src_box = box(*src_grid_subset.extent_global)
                        if not does_contain(src_box, dst_box):
                            raise ValueError('Contains check failed.')

                    # Try to reduce the coordinates in the case of unstructured grid data.
                    if hasattr(src_grid_subset, 'reduce_global'):
                        src_grid_subset = src_grid_subset.reduce_global()
                else:
                    src_grid_subset = VariableCollection(is_empty=True)

                if src_grid_subset.is_empty:
                    src_grid_slice = None
                else:
                    src_grid_slice = {src_grid_subset.dimensions[ii].name: src_grid_slice[ii] for ii in
                                      range(src_grid_subset.ndim)}

            if yield_dst:
                yld = (src_grid_subset, src_grid_slice, dst_grid_subset, dst_slice)
            else:
                yld = src_grid_subset, src_grid_slice

            yield yld

    def write_subsets(self):
        """
        Write grid subsets to netCDF files using the provided filename templates.
        """
        src_filenames = []
        dst_filenames = []
        wgt_filenames = []
        dst_slices = []
        src_slices = []
        index_path = self.create_full_path_from_template('index_file')

        # nzeros = len(str(reduce(lambda x, y: x * y, self.nsplits_dst)))

        ctr = 1
        for sub_src, src_slc, sub_dst, dst_slc in self.iter_src_grid_subsets(yield_dst=True):
            # if vm.rank == 0:
            #     vm.rank_print('write_subset iterator count :: {}'.format(ctr))
            #     tstart = time.time()
            # padded = create_zero_padded_integer(ctr, nzeros)

            src_path = self.create_full_path_from_template('src_template', index=ctr)
            dst_path = self.create_full_path_from_template('dst_template', index=ctr)
            wgt_path = self.create_full_path_from_template('wgt_template', index=ctr)

            src_filenames.append(os.path.split(src_path)[1])
            dst_filenames.append(os.path.split(dst_path)[1])
            wgt_filenames.append(wgt_path)
            dst_slices.append(dst_slc)
            src_slices.append(src_slc)

            # Only write destinations if an iterator is not provided.
            if self.iter_dst is None:
                zip_args = [[sub_src, sub_dst], [src_path, dst_path]]
            else:
                zip_args = [[sub_src], [src_path]]

            for target, path in zip(*zip_args):
                with vm.scoped_by_emptyable('field.write', target):
                    if not vm.is_null:
                        ocgis_lh(msg='writing: {}'.format(path), level=logging.DEBUG)
                        field = Field(grid=target)
                        field.write(path)
                        ocgis_lh(msg='finished writing: {}'.format(path), level=logging.DEBUG)

            # Increment the counter outside of the loop to avoid counting empty subsets.
            ctr += 1

            # if vm.rank == 0:
            #     tstop = time.time()
            #     vm.rank_print('timing::write_subset iteration::{}'.format(tstop - tstart))

        # Global shapes require a VM global scope to collect.
        src_global_shape = global_grid_shape(self.src_grid)
        dst_global_shape = global_grid_shape(self.dst_grid)

        # Gather and collapse source slices as some may be empty and we write on rank 0.
        gathered_src_grid_slice = vm.gather(src_slices)
        if vm.rank == 0:
            len_src_slices = len(src_slices)
            new_src_grid_slice = [None] * len_src_slices
            for idx in range(len_src_slices):
                for rank_src_grid_slice in gathered_src_grid_slice:
                    if rank_src_grid_slice[idx] is not None:
                        new_src_grid_slice[idx] = rank_src_grid_slice[idx]
                        break
            src_slices = new_src_grid_slice

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
                vidx.attrs[GridSplitterConstants.IndexFile.NAME_SRC_GRID_SHAPE] = src_global_shape
                vidx.attrs[GridSplitterConstants.IndexFile.NAME_DST_GRID_SHAPE] = dst_global_shape

                vc.add_variable(vidx)

                for idx in range(len(vname)):
                    v = Variable(name=vname[idx], dimensions=dim, dtype=str, value=values[idx], attrs=attrs[idx])
                    vc.add_variable(v)

                bounds_dimension = Dimension(name='bounds', size=2)
                # TODO: This needs to work with four dimensions.
                # Source -----------------------------------------------------------------------------------------------
                self.src_grid._gs_create_index_bounds_(RegriddingRole.SOURCE, vidx, vc, src_slices, dim,
                                                       bounds_dimension)

                # Destination ------------------------------------------------------------------------------------------
                self.dst_grid._gs_create_index_bounds_(RegriddingRole.DESTINATION, vidx, vc, dst_slices, dim,
                                                       bounds_dimension)

                vc.write(index_path)

        vm.barrier()

    def _gs_remap_weight_variable_(self, ii, wvn, odata, src_indices, dst_indices, ifile, gidx,
                                   split_grids_directory=None):
        if wvn == 'S':
            pass
        else:
            ifc = GridSplitterConstants.IndexFile
            if wvn == 'row':
                is_unstruct = isinstance(self.dst_grid, GridUnstruct)
                if is_unstruct:
                    dst_filename = ifile[gidx[ifc.NAME_DESTINATION_VARIABLE]].join_string_value()[ii]
                    dst_filename = os.path.join(split_grids_directory, dst_filename)
                    oindices = RequestDataset(dst_filename).get()[ifc.NAME_DSTIDX_GUID].get_value()
                else:
                    y_bounds = ifile[gidx[ifc.NAME_Y_DST_BOUNDS_VARIABLE]].get_value()
                    x_bounds = ifile[gidx[ifc.NAME_X_DST_BOUNDS_VARIABLE]].get_value()
                    indices = dst_indices
            elif wvn == 'col':
                is_unstruct = isinstance(self.src_grid, GridUnstruct)
                if is_unstruct:
                    src_filename = ifile[gidx[ifc.NAME_SOURCE_VARIABLE]].join_string_value()[ii]
                    src_filename = os.path.join(split_grids_directory, src_filename)
                    oindices = RequestDataset(src_filename).get()[ifc.NAME_SRCIDX_GUID].get_value()
                else:
                    y_bounds = ifile[gidx[ifc.NAME_Y_SRC_BOUNDS_VARIABLE]].get_value()
                    x_bounds = ifile[gidx[ifc.NAME_X_SRC_BOUNDS_VARIABLE]].get_value()
                    indices = src_indices
            else:
                raise NotImplementedError
            if not is_unstruct:
                islice = [slice(y_bounds[ii][0], y_bounds[ii][1]),
                          slice(x_bounds[ii][0], x_bounds[ii][1])]
                oindices = indices[islice]
                oindices = oindices.flatten()
            odata = oindices[odata - 1]

        return odata


def create_slice_from_tuple(tup):
    return slice(tup[0], tup[1])


def create_slices_for_dimension(size, splits):
    ompi = OcgDist(size=splits)
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


def global_grid_shape(grid):
    with vm.scoped_by_emptyable('global grid shape', grid):
        if not vm.is_null:
            return grid.shape_global


def is_unstructured(target):
    return isinstance(target, GridUnstruct)
