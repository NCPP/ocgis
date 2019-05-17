import logging
import os

import netCDF4 as nc
import numpy as np
from ocgis import constants
from ocgis.base import AbstractOcgisObject, grid_abstraction_scope
from ocgis.collection.field import Field
from ocgis.constants import GridChunkerConstants, RegriddingRole, Topology
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.base import iter_spatial_decomposition
from ocgis.spatial.geomc import AbstractGeometryCoordinates
from ocgis.spatial.grid import GridUnstruct, AbstractGrid, Grid
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import VariableCollection, Variable
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.vmachine.core import vm
from ocgis.vmachine.mpi import redistribute_by_src_idx
from shapely.geometry import box


class GridChunker(AbstractOcgisObject):
    """
    Splits source and destination grids into separate netCDF files. "Source" is intended to mean the source data for a
    regridding operation. "Destination" is the destination grid for regridding operation.

    The destination subset extents are buffered to ensure full overlap with the source destination subset. Hence,
    elements in the destination subset are globally unique and source subset elements are not necessarily globally
    unique.

    .. note:: Grid parent variable collections may be altered during initializations to account for global source
     indexing.

    .. note:: All function calls are collective.

    :param source: The source object for a regridding operation. The object must either be a grid or an object from
     which a grid is retrievable.
    :type source: :class:`~ocgis.spatial.grid.AbstractGrid` | :class:`~ocgis.RequestDataset` | :class:`~ocgis.Field`
    :param destination: The destination object for a regridding operation. The object must either be a grid or an object
     from which a grid is retrievable.
    :type destination: :class:`~ocgis.spatial.grid.AbstractGrid` | :class:`~ocgis.RequestDataset` | :class:`~ocgis.Field`
    :param tuple nchunks_dst: The split count for the grid. Tuple length must match the dimension count of the grid.

    >>> nchunks_dst = (2, 3)

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
    :param bool genweights: If ``False``, do no generate regridding weight files using ESMF. If ``True``, generate
     regridding weight files for each source-destination chunk.
    :param dict esmf_kwargs: Optional overloads for keyword arguments to ESMF interfaces. Currently supported keyword
     arguments are below.

    ======================= ============== ===============================================================
    Name                    Default        Possible
    ======================= ============== ===============================================================
    ``'regrid_method'``     ``'CONSERVE'`` ``'CONSERVE'``, ``'BILINEAR'``, ``'PATCH'``, ``'NEAREST_STOD'``
    ``'unmapped_action'``   ``'IGNORE'``   ``'IGNORE'``, ``'ERROR'``
    ``'ignore_degenerate'`` ``False``      ``True``/``False``
    ======================= ============== ===============================================================

    :param bool use_spatial_decomp: If ``True``, use a spatial decomposition as opposed to an index-based decomposition
     when creating destination chunks. A spatial decomposition ensures destination coordinates are spatially "clumped"
     and is recommended for unstructured datasets. If ``'auto'``, choose the best approach from the grid type.
    :param bool eager: If ``True``, load grid data from disk before chunking. This avoids always loading the data from
     disk for sourced datasets following a subset. There will be an improvement in performance but an increase in the
     memory used.
    :raises: ValueError
    """

    def __init__(self, source, destination, nchunks_dst=None, paths=None, check_contains=False, allow_masked=True,
                 src_grid_resolution=None, dst_grid_resolution=None, optimized_bbox_subset='auto', iter_dst=None,
                 buffer_value=None, redistribute=False, genweights=False, esmf_kwargs=None, use_spatial_decomp='auto',
                 eager=True):
        self._src_grid = None
        self._dst_grid = None
        self._buffer_value = None
        self._nchunks_dst = None
        self._optimized_bbox_subset = None
        self._use_spatial_decomp = use_spatial_decomp

        self.genweights = genweights
        self.source = source
        self.destination = destination
        self.eager = eager

        if esmf_kwargs is None:
            esmf_kwargs = {}
        if self.genweights:
            esmf_kwargs = esmf_kwargs.copy()
            from ocgis.regrid.base import update_esmf_kwargs
            update_esmf_kwargs(esmf_kwargs)
        self.esmf_kwargs = esmf_kwargs

        self.nchunks_dst = nchunks_dst
        self.check_contains = check_contains
        self.allow_masked = allow_masked
        self.src_grid_resolution = src_grid_resolution
        self.dst_grid_resolution = dst_grid_resolution
        self.iter_dst = iter_dst
        self.buffer_value = buffer_value
        self.optimized_bbox_subset = optimized_bbox_subset
        self.redistribute = redistribute

        # Call each grid's grid splitter initialize routine.
        self.src_grid._gc_initialize_(RegriddingRole.SOURCE)
        self.dst_grid._gc_initialize_(RegriddingRole.DESTINATION)

        # Construct default paths if None are provided.
        defaults = constants.GridChunkerConstants.DEFAULT_PATHS
        if paths is None:
            paths = defaults
        else:
            for k, v in defaults.items():
                if k not in paths:
                    paths[k] = v
        if 'wd' not in paths:
            paths['wd'] = os.getcwd()
        self.paths = paths

    @property
    def buffer_value(self):
        """
        Spatial distance in units of the destination grid to buffer the destination grid chunk's spatial extent when
        subsetting the associated source grid. Defaults to the higher spatial resolution times a modifier
        (:attr:`ocgis.constants.GridChunkerConstants.BUFFER_RESOLUTION_MODIFIER`).

        :param float value: Spatial buffer value
        :rtype: float
        """
        if self._buffer_value is None:
            try:
                if self.dst_grid_resolution is None:
                    dst_grid_resolution = self.dst_grid.resolution_max
                else:
                    dst_grid_resolution = self.dst_grid_resolution
                if self.src_grid_resolution is None:
                    src_grid_resolution = self.src_grid.resolution_max
                else:
                    src_grid_resolution = self.src_grid_resolution

                if dst_grid_resolution <= src_grid_resolution:
                    target_resolution = dst_grid_resolution
                else:
                    target_resolution = src_grid_resolution
                ret = GridChunkerConstants.BUFFER_RESOLUTION_MODIFIER * target_resolution
            except NotImplementedError:
                # Unstructured grids do not have an associated resolution unless they are isomorphic.
                if isinstance(self.src_grid, GridUnstruct) or isinstance(self.dst_grid, GridUnstruct):
                    ret = None
                else:
                    raise
        else:
            ret = self._buffer_value
        return ret

    @buffer_value.setter
    def buffer_value(self, value):
        self._buffer_value = value

    @property
    def nchunks_dst(self):
        if self._nchunks_dst is None:
            ret = self.dst_grid._gc_nchunks_dst_(self)
        else:
            ret = self._nchunks_dst
        return ret

    @nchunks_dst.setter
    def nchunks_dst(self, value):
        # Assert the split dimension matches the destination grid dimension.
        if value is not None and len(value) != self.dst_grid.ndim:
            raise ValueError('The number of splits must match the grid dimension count.')
        self._nchunks_dst = value

    @property
    def dst_grid(self):
        """
        Get the destination grid.

        :return: :class:`~ocgis.spatial.grid.AbstractGrid`
        """
        if self._dst_grid is None:
            self._dst_grid = get_grid_object(self.destination)
        return self._dst_grid

    @property
    def optimized_bbox_subset(self):
        """
        If ``True``, use an optimized bounding box subset to spatially subset the source grid.

        :param value: If ``'auto'``, choose the optimization based on grid isomorphism.
        :type value: str | bool
        :rtype: bool
        """
        if self._optimized_bbox_subset == 'auto':
            if (self.src_grid_resolution is not None and self.dst_grid_resolution is not None) or (
                    self.src_grid.resolution_max is not None or self.src_grid_resolution is not None) and (
                    self.dst_grid.resolution_max is not None or self.dst_grid_resolution is not None):
                ret = True
            else:
                ret = False
        else:
            ret = self._optimized_bbox_subset
        return ret

    @optimized_bbox_subset.setter
    def optimized_bbox_subset(self, value):
        assert value is not None
        self._optimized_bbox_subset = value

    @property
    def src_grid(self):
        """
        Get the source grid.

        :return: :class:`~ocgis.spatial.grid.AbstractGrid`
        """
        if self._src_grid is None:
            self._src_grid = get_grid_object(self.source)
        return self._src_grid

    @property
    def use_spatial_decomp(self):
        ref = self._use_spatial_decomp
        if ref == 'auto':
            if isinstance(self.dst_grid, Grid):
                ref = False
            else:
                ref = True
        return ref

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
        ifc = GridChunkerConstants.IndexFile
        gidx = ifile[ifc.NAME_INDEX_VARIABLE].attrs

        src_global_shape = gidx[ifc.NAME_SRC_GRID_SHAPE]
        dst_global_shape = gidx[ifc.NAME_DST_GRID_SHAPE]

        # Get the global weight dimension size.
        n_s_size = 0
        weight_filename = ifile[gidx[ifc.NAME_WEIGHTS_VARIABLE]]
        wv = weight_filename.join_string_value()
        split_weight_file_directory = self.paths['wd']
        for wfn in map(lambda x: os.path.join(split_weight_file_directory, os.path.split(x)[1]), wv):
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
        src_indices = self.src_grid._gc_create_global_indices_(src_global_shape)
        dst_indices = self.dst_grid._gc_create_global_indices_(dst_global_shape)

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
                    odata = self._gc_remap_weight_variable_(ii, wvn, odata, src_indices, dst_indices, ifile, gidx,
                                                            split_grids_directory=split_grids_directory)
                except IndexError as e:
                    msg = "Weight filename: '{}'; Weight Variable Name: '{}'. {}".format(wfn, wvn, str(e))
                    raise IndexError(msg)
                out_wds[wvn][sidx:sidx + odata.size] = odata
                out_wds.sync()
            sidx += odata.size
        out_wds.close()

    @staticmethod
    def insert_weighted(index_path, dst_wd, dst_master_path, data_variables='auto'):
        """
        Inserted weighted, destination variable data into the master destination file.

        :param str index_path: Path to the split index netCDF file.
        :param str dst_wd: Working directory containing the destination files holding the weighted data.
        :param str dst_master_path: Path to the destination master weight file.
        :param list data_variables: Optional list of data variables. Otherwise, auto-discovery is used.
        """
        if vm.size > 1:
            raise NotImplementedError('serial only')

        index_field = RequestDataset(index_path).get()
        gs_index_v = index_field[GridChunkerConstants.IndexFile.NAME_INDEX_VARIABLE]
        dst_filenames = gs_index_v.attrs[GridChunkerConstants.IndexFile.NAME_DESTINATION_VARIABLE]
        dst_filenames = index_field[dst_filenames]

        y_bounds = GridChunkerConstants.IndexFile.NAME_Y_DST_BOUNDS_VARIABLE
        y_bounds = gs_index_v.attrs[y_bounds]
        y_bounds = index_field[y_bounds].get_value()

        x_bounds = GridChunkerConstants.IndexFile.NAME_X_DST_BOUNDS_VARIABLE
        x_bounds = gs_index_v.attrs[x_bounds]
        x_bounds = index_field[x_bounds].get_value()

        joined = dst_filenames.join_string_value()
        if data_variables == 'auto':
            v = None
        else:
            v = data_variables

        dst_master_field = RequestDataset(dst_master_path, variable=v).get()
        for data_variable in dst_master_field.data_variables:
            assert not data_variable.has_allocated_value
            if data_variable.ndim == 3:
                for time_index in range(dst_master_field.time.shape[0]):
                    for vidx, source_path in enumerate(joined):
                        source_path = os.path.join(dst_wd, source_path)
                        slc = {dst_master_field.time.dimensions[0].name: time_index,
                               dst_master_field.y.dimensions[0].name: slice(None),
                               dst_master_field.x.dimensions[0].name: slice(None)}
                        source_field = RequestDataset(source_path).create_field()
                        try:
                            source_data = source_field[data_variable.name][slc]
                        except KeyError:
                            if data_variable.name not in source_field.keys():
                                msg = "The destination variable '{}' is not in the destination file '{}'. Was SMM applied?".format(
                                    data_variable.name, source_path)
                                raise KeyError(msg)
                            else:
                                raise
                        assert not source_data.has_allocated_value
                        with nc.Dataset(dst_master_path, 'a') as ds:
                            ds.variables[data_variable.name][time_index, y_bounds[vidx][0]:y_bounds[vidx][1],
                            x_bounds[vidx][0]:x_bounds[vidx][1]] = source_data.get_value()
            elif data_variable.ndim == 2:
                for vidx, source_path in enumerate(joined):
                    source_path = os.path.join(dst_wd, source_path)
                    source_data = RequestDataset(source_path).get()[data_variable.name]
                    assert not source_data.has_allocated_value
                    with nc.Dataset(dst_master_path, 'a') as ds:
                        ds.variables[data_variable.name][y_bounds[vidx][0]:y_bounds[vidx][1],
                        x_bounds[vidx][0]:x_bounds[vidx][1]] = source_data.get_value()
            else:
                raise NotImplementedError(data_variable.ndim)

    def iter_dst_grid_slices(self, yield_idx=None):
        """
        Yield global slices for the destination grid using guidance from ``nchunks_dst``.

        :param int yield_idx: If a zero-based integer, only yield for this chunk index and skip everything else.
        :return: A dictionary with keys as the grid dimension names and the values the associated slice for that
         dimension.
        :rtype: dict

        >>> example_yield = {'dimx': slice(2, 4), 'dimy': slice(10, 20)}
        """
        return self.dst_grid._gc_iter_dst_grid_slices_(self, yield_idx=yield_idx)

    def iter_dst_grid_subsets(self, yield_slice=False, yield_idx=None):
        """
        Yield destination grid subsets.

        :param int yield_idx: If a zero-based integer, only yield for this chunk index and skip everything else.
        :param bool yield_slice: If ``True``, yield the slice used on the destination grid.
        :rtype: :class:`ocgis.spatial.grid.AbstractGrid`
        """
        if self.use_spatial_decomp:
            for sub, slc in iter_spatial_decomposition(self.dst_grid, self.nchunks_dst, optimized_bbox_subset=True,
                                                       yield_idx=yield_idx):
                if yield_slice:
                    # Spatial subset may be empty on a rank...
                    if sub.is_empty:
                        slc = None
                    else:
                        slc = {d.name: slc[ii] for ii, d in enumerate(sub.dimensions)}

                    yield sub, slc
                else:
                    yield sub
        else:
            for slc in self.iter_dst_grid_slices(yield_idx=yield_idx):
                sub = self.dst_grid.get_distributed_slice(slc)
                if yield_slice:
                    yield sub, slc
                else:
                    yield sub

    def iter_src_grid_subsets(self, yield_dst=False, yield_idx=None):
        """
        Yield source grid subset using the extent of its associated destination grid subset.

        :param bool yield_dst: If ``True``, yield the destination subset as well as the source grid subset.
        :param int yield_idx: If a zero-based integer, only yield for this chunk index and skip everything else.
        :rtype: tuple(:class:`ocgis.spatial.grid.AbstractGrid`, `slice-like`)
        """
        if yield_dst:
            yield_slice = True
        else:
            yield_slice = False

        buffer_value = self.buffer_value

        dst_grid_wrapped_state = self.dst_grid.wrapped_state
        dst_grid_crs = self.dst_grid.crs

        # Use a destination grid iterator if provided.
        if self.iter_dst is not None:
            iter_dst = self.iter_dst(self, yield_slice=yield_slice, yield_idx=yield_idx)
        else:
            iter_dst = self.iter_dst_grid_subsets(yield_slice=yield_slice, yield_idx=yield_idx)

        # Loop over each destination grid subset.
        ocgis_lh(logger='grid_chunker', msg='starting "for yld in iter_dst"', level=logging.DEBUG)
        for iter_dst_ctr, yld in enumerate(iter_dst, start=1):
            ocgis_lh(msg=["iter_dst_ctr", iter_dst_ctr], level=logging.DEBUG)
            if yield_slice:
                dst_grid_subset, dst_slice = yld
            else:
                dst_grid_subset = yld

            # All masked destinations are very problematic for ESMF
            with vm.scoped_by_emptyable('global mask', dst_grid_subset):
                if not vm.is_null:
                    if dst_grid_subset.has_mask_global:
                        if dst_grid_subset.has_mask and dst_grid_subset.has_masked_values:
                            all_masked = dst_grid_subset.get_mask().all()
                        else:
                            all_masked = False
                        all_masked_gather = vm.gather(all_masked)
                        if vm.rank == 0:
                            if all(all_masked_gather):
                                exc = ValueError("Destination subset all masked")
                                try:
                                    raise exc
                                finally:
                                    vm.abort(exc=exc)

            dst_box = None
            with vm.scoped_by_emptyable('extent_global', dst_grid_subset):
                if not vm.is_null:
                    # Use the extent of the polygon for determining the bounding box. This ensures conservative
                    # regridding will be fully mapped.
                    if isinstance(dst_grid_subset, AbstractGeometryCoordinates):
                        target_grid = dst_grid_subset.parent.grid
                    else:
                        target_grid = dst_grid_subset

                    # Try to reduce the coordinates in the case of unstructured grid data. Ensure the data also has a
                    # coordinate index. SCRIP grid files, for example, do not have a coordinate index like UGRID.
                    if hasattr(target_grid, 'reduce_global') and Topology.POLYGON in target_grid.abstractions_available and target_grid.cindex is not None:
                        ocgis_lh(logger='grid_chunker', msg='starting reduce_global for dst_grid_subset',
                                 level=logging.DEBUG)
                        target_grid = target_grid.reduce_global()
                        ocgis_lh(logger='grid_chunker', msg='finished reduce_global for dst_grid_subset',
                                 level=logging.DEBUG)

                    extent_global = target_grid.parent.attrs.get('extent_global')
                    if extent_global is None:
                        with grid_abstraction_scope(target_grid, Topology.POLYGON):
                            extent_global = target_grid.extent_global

                    if self.check_contains:
                        dst_box = box(*target_grid.extent_global)

                    sub_box = box(*extent_global)
                    if buffer_value is not None:
                        # Use the envelope! A buffer returns "fancy" borders. We just want to expand the bounding box.
                        sub_box = sub_box.buffer(buffer_value).envelope

                    ocgis_lh(msg=str(sub_box.bounds), level=logging.DEBUG, logger='grid_chunker')
                else:
                    sub_box, dst_box = [None, None]

            live_ranks = vm.get_live_ranks_from_object(dst_grid_subset)
            sub_box = vm.bcast(sub_box, root=live_ranks[0])

            if self.check_contains:
                dst_box = vm.bcast(dst_box, root=live_ranks[0])

            sub_box = GeometryVariable.from_shapely(sub_box, is_bbox=True, wrapped_state=dst_grid_wrapped_state,
                                                    crs=dst_grid_crs)
            ocgis_lh(logger='grid_chunker', msg='starting "self.src_grid.get_intersects"', level=logging.DEBUG)
            src_grid_subset, src_grid_slice = self.src_grid.get_intersects(sub_box, keep_touches=False, cascade=False,
                                                                           optimized_bbox_subset=self.optimized_bbox_subset,
                                                                           return_slice=True)
            ocgis_lh(logger='grid_chunker', msg='finished "self.src_grid.get_intersects"', level=logging.DEBUG)

            # Reload the data using a new source index distribution.
            if hasattr(src_grid_subset, 'reduce_global') and src_grid_subset.cindex is not None:
                # Only redistribute if we have one live rank.
                if self.redistribute and len(vm.get_live_ranks_from_object(src_grid_subset)) > 0:
                    ocgis_lh(logger='grid_chunker', msg='starting redistribute', level=logging.DEBUG)
                    topology = src_grid_subset.abstractions_available[Topology.POLYGON]
                    cindex = topology.cindex
                    redist_dimname = self.src_grid.abstractions_available[Topology.POLYGON].element_dim.name
                    if src_grid_subset.is_empty:
                        redist_dim = None
                    else:
                        redist_dim = topology.element_dim
                    redistribute_by_src_idx(cindex, redist_dimname, redist_dim)
                    ocgis_lh(logger='grid_chunker', msg='finished redistribute', level=logging.DEBUG)

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
                    if hasattr(src_grid_subset, 'reduce_global') and src_grid_subset.cindex is not None:
                        ocgis_lh(logger='grid_chunker', msg='starting reduce_global', level=logging.DEBUG)
                        src_grid_subset = src_grid_subset.reduce_global()
                        ocgis_lh(logger='grid_chunker', msg='finished reduce_global', level=logging.DEBUG)
                else:
                    pass
                    # src_grid_subset = VariableCollection(is_empty=True)

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

    @staticmethod
    def smm(*args, **kwargs):
        """See :meth:`ocgis.regrid.base.smm`"""
        from ocgis.regrid.base import smm
        smm(*args, **kwargs)

    def write_chunks(self):
        """
        Write grid subsets to netCDF files using the provided filename templates. This will also generate ESMF
        regridding weights for each subset if requested.
        """
        src_filenames = []
        dst_filenames = []
        wgt_filenames = []
        dst_slices = []
        src_slices = []
        index_path = self.create_full_path_from_template('index_file')

        # nzeros = len(str(reduce(lambda x, y: x * y, self.nchunks_dst)))

        ctr = 1
        ocgis_lh(logger='grid_chunker', msg='starting self.iter_src_grid_subsets', level=logging.DEBUG)
        for sub_src, src_slc, sub_dst, dst_slc in self.iter_src_grid_subsets(yield_dst=True):
            ocgis_lh(logger='grid_chunker', msg='finished iteration {} for self.iter_src_grid_subsets'.format(ctr),
                     level=logging.DEBUG)

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

            cc = 1
            for target, path in zip(*zip_args):
                with vm.scoped_by_emptyable('field.write' + str(cc), target):
                    if not vm.is_null:
                        ocgis_lh(logger='grid_chunker', msg='write_chunks:writing: {}'.format(path),
                                 level=logging.DEBUG)
                        field = Field(grid=target)
                        field.write(path)
                        ocgis_lh(logger='grid_chunker', msg='write_chunks:finished writing: {}'.format(path),
                                 level=logging.DEBUG)
                cc += 1

            # Increment the counter outside of the loop to avoid counting empty subsets.
            ctr += 1

            # Generate an ESMF weights file if requested and at least one rank has data on it.
            if self.genweights and len(vm.get_live_ranks_from_object(sub_src)) > 0:
                vm.barrier()
                self.write_esmf_weights(src_path, dst_path, wgt_path, src_grid=sub_src, dst_grid=sub_dst)
                vm.barrier()

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
                grid_chunker_destination = GridChunkerConstants.IndexFile.NAME_DESTINATION_VARIABLE
                attrs = [{'esmf_role': 'grid_chunker_source'},
                         {'esmf_role': grid_chunker_destination},
                         {'esmf_role': 'grid_chunker_weights'}]

                vc = VariableCollection()

                grid_chunker_index = GridChunkerConstants.IndexFile.NAME_INDEX_VARIABLE
                vidx = Variable(name=grid_chunker_index)
                vidx.attrs['esmf_role'] = grid_chunker_index
                vidx.attrs['grid_chunker_source'] = 'source_filename'
                vidx.attrs[GridChunkerConstants.IndexFile.NAME_DESTINATION_VARIABLE] = 'destination_filename'
                vidx.attrs['grid_chunker_weights'] = 'weights_filename'
                vidx.attrs[GridChunkerConstants.IndexFile.NAME_SRC_GRID_SHAPE] = src_global_shape
                vidx.attrs[GridChunkerConstants.IndexFile.NAME_DST_GRID_SHAPE] = dst_global_shape

                vc.add_variable(vidx)

                for idx in range(len(vname)):
                    v = Variable(name=vname[idx], dimensions=dim, dtype=str, value=values[idx], attrs=attrs[idx])
                    vc.add_variable(v)

                bounds_dimension = Dimension(name='bounds', size=2)
                # TODO: This needs to work with four dimensions.
                # Source -----------------------------------------------------------------------------------------------
                self.src_grid._gc_create_index_bounds_(RegriddingRole.SOURCE, vidx, vc, src_slices, dim,
                                                       bounds_dimension)

                # Destination ------------------------------------------------------------------------------------------
                self.dst_grid._gc_create_index_bounds_(RegriddingRole.DESTINATION, vidx, vc, dst_slices, dim,
                                                       bounds_dimension)

                vc.write(index_path)

        vm.barrier()

    def write_esmf_weights(self, src_path, dst_path, wgt_path, src_grid=None, dst_grid=None):
        """
        Write ESMF regridding weights for a source and destination filename combination.

        :param str src_path: Full path to source file
        :param str dst_path: Full path to destination file
        :param str wgt_path: Path to output weight file
        :param src_grid: If provided, use this source grid for identifying ESMF parameters
        :type src_grid: :class:`ocgis.spatial.grid.AbstractGrid`
        :param dst_grid: If provided, use this destination grid for identifying ESMF parameters
        :type dst_grid: :class:`ocgis.spatial.grid.AbstractGrid`
        """
        from ocgis.regrid.base import create_esmf_field, create_esmf_regrid
        if src_grid is None:
            src_grid = self.src_grid
        if dst_grid is None:
            dst_grid = self.dst_grid

        assert wgt_path is not None

        srcfield, srcgrid = create_esmf_field(src_path, src_grid, self.esmf_kwargs)
        dstfield, dstgrid = create_esmf_field(dst_path, dst_grid, self.esmf_kwargs)
        regrid = None

        try:
            regrid = create_esmf_regrid(srcfield=srcfield, dstfield=dstfield, filename=wgt_path, **self.esmf_kwargs)
        finally:
            to_destroy = [regrid, srcgrid, srcfield, dstgrid, dstfield]
            for t in to_destroy:
                if t is not None:
                    t.destroy()
            del regrid
            del srcgrid
            del srcfield
            del dstgrid
            del dstfield

    def _gc_remap_weight_variable_(self, ii, wvn, odata, src_indices, dst_indices, ifile, gidx,
                                   split_grids_directory=None):
        if wvn == 'S':
            pass
        else:
            ifc = GridChunkerConstants.IndexFile
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
                islice = tuple([slice(y_bounds[ii][0], y_bounds[ii][1]),
                                slice(x_bounds[ii][0], x_bounds[ii][1])])
                oindices = indices[islice]
                oindices = oindices.flatten()

            odata = oindices[odata - 1]

        return odata


def does_contain(container, containee):
    intersection = container.intersection(containee)
    return np.isclose(intersection.area, containee.area)


def get_grid_object(obj, load=True):
    if isinstance(obj, AbstractGrid):
        res = obj
    elif isinstance(obj, RequestDataset):
        res = obj.create_field().grid
    elif isinstance(obj, Field):
        res = obj.grid
    else:
        raise NotImplementedError(obj)

    if load:
        res.parent.load()

    return res


def global_grid_shape(grid):
    with vm.scoped_by_emptyable('global grid shape', grid):
        if not vm.is_null:
            return grid.shape_global
