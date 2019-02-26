#!/usr/bin/env python

import logging
import os
import shutil
import tempfile

import click
import ocgis
from ocgis import RequestDataset, GeometryVariable
from ocgis.base import grid_abstraction_scope
from ocgis.constants import DriverKey, Topology, GridChunkerConstants
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.spatial.spatial_subset import SpatialSubsetOperation
from ocgis.util.logging_ocgis import ocgis_lh
from shapely.geometry import box


@click.group()
def ocli():
    pass


@ocli.command(help='Generate regridding weights using a spatial decomposition.', name='chunked-rwg')
@click.option('-s', '--source', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to the source grid NetCDF file.')
@click.option('-d', '--destination', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to the destination grid NetCDF file.')
@click.option('-n', '--nchunks_dst',
              help='Single integer or sequence defining the chunking decomposition for the destination grid. Each '
                   'value is the number of chunks along each decomposed axis. For unstructured grids, provide a single '
                   'value (i.e. 100). For logically rectangular grids, two values are needed to describe the x and y '
                   'decomposition (i.e. 10,20). Required if --genweights and not --spatial_subset.')
@click.option('--merge/--no_merge', default=True,
              help='(default=merge) If --merge, merge weight file chunks into a global weight file.')
@click.option('-w', '--weight', required=False, type=click.Path(exists=False, dir_okay=False),
              help='Path to the output global weight file. Required if --merge.')
@click.option('--esmf_src_type', type=str, nargs=1, default='GRIDSPEC',
              help='(default=GRIDSPEC) ESMF source grid type. Supports GRIDSPEC, UGRID, and SCRIP.')
@click.option('--esmf_dst_type', type=str, nargs=1, default='GRIDSPEC',
              help='(default=GRIDSPEC) ESMF destination grid type. Supports GRIDSPEC, UGRID, and SCRIP.')
@click.option('--genweights/--no_genweights', default=True,
              help='(default=True) Generate weights using ESMF for each source and destination subset.')
@click.option('--esmf_regrid_method', type=str, nargs=1, default='CONSERVE',
              help='(default=CONSERVE) The ESMF regrid method. Only applicable with --genweights. Supports CONSERVE, '
                   'BILINEAR. PATCH, and NEAREST_STOD.')
@click.option('--spatial_subset/--no_spatial_subset', default=False,
              help='(default=no_spatial_subset) Optionally subset the destination grid by the bounding box spatial '
                   'extent of the source grid. This will not work in parallel if --genweights.')
@click.option('--src_resolution', type=float, nargs=1,
              help='Optionally overload the spatial resolution of the source grid. If provided, assumes an isomorphic '
                   'structure. Spatial resolution is the mean distance between grid cell center coordinates.')
@click.option('--dst_resolution', type=float, nargs=1,
              help='Optionally overload the spatial resolution of the destination grid. If provided, assumes an '
                   'isomorphic structure. Spatial resolution is the mean distance between grid cell center '
                   'coordinates.')
@click.option('--buffer_distance', type=float, nargs=1,
              help='Optional spatial buffer distance (in units of the destination grid coordinates) to use when '
                   'subsetting the source grid by the spatial extent of a destination grid or chunk. This is computed '
                   'internally if not provided. Useful to override if the area of influence for a source-destination '
                   'mapping is known a priori.')
@click.option('--wd', type=click.Path(exists=False), default=None,
              help="Optional working directory for intermediate chunk files. Creates a directory in the system's "
                   "temporary scratch space if not provided.")
@click.option('--persist/--no_persist', default=False,
              help='(default=no_persist) If --persist, do not remove the working directory --wd following execution.')
@click.option('--eager/--not_eager', default=True,
              help='(default=eager) If --eager, load all data from the grids into memory before subsetting. This will '
                   'increase performance as loading data for each chunk is avoided. Set this to --not_eager for a more '
                   'memory efficient execution at the expense of additional IO operations.')
@click.option('--ignore_degenerate/--no_ignore_degenerate', default=False,
              help='(default=no_ignore_degenerate) If --ignore_degenerate, skip degenerate coordinates when regridding '
                   'and do not raise an exception.')
def chunked_rwg(source, destination, weight, nchunks_dst, merge, esmf_src_type, esmf_dst_type, genweights,
                esmf_regrid_method, spatial_subset, src_resolution, dst_resolution, buffer_distance, wd, persist,
                eager, ignore_degenerate):
    if not ocgis.env.USE_NETCDF4_MPI:
        msg = ('env.USE_NETCDF4_MPI is False. Considerable performance gains are possible if this is True. Is '
               'netCDF4-python built with parallel support?')
        ocgis_lh(msg, level=logging.WARN, logger='ocli.chunked-rwg', force=True)

    if nchunks_dst is not None:
        # Format the chunking decomposition from its string representation.
        if ',' in nchunks_dst:
            nchunks_dst = nchunks_dst.split(',')
        else:
            nchunks_dst = [nchunks_dst]
        nchunks_dst = tuple([int(ii) for ii in nchunks_dst])
    if merge:
        if not spatial_subset and weight is None:
            raise ValueError('"weight" must be a valid path if --merge')
    if spatial_subset and genweights and weight is None:
        raise ValueError('"weight" must be a valid path if --genweights')

    # Make a temporary working directory is one is not provided by the client. Only do this if we are writing subsets
    # and it is not a merge only operation.
    if wd is None:
        if ocgis.vm.rank == 0:
            wd = tempfile.mkdtemp(prefix='ocgis_chunked_rwg_')
        wd = ocgis.vm.bcast(wd)
    else:
        if ocgis.vm.rank == 0:
            # The working directory must not exist to proceed.
            if os.path.exists(wd):
                raise ValueError("Working directory 'wd' must not exist.")
            else:
                # Make the working directory nesting as needed.
                os.makedirs(wd)
        ocgis.vm.barrier()

    if merge and not spatial_subset or (spatial_subset and genweights):
        if _is_subdir_(wd, weight):
            raise ValueError(
                'Merge weight file path must not in the working directory. It may get unintentionally deleted with the --no_persist flag.')

    # Create the source and destination request datasets.
    rd_src = _create_request_dataset_(source, esmf_src_type)
    rd_dst = _create_request_dataset_(destination, esmf_dst_type)

    # Execute a spatial subset if requested.
    paths = None
    if spatial_subset:
        # TODO: This path should be customizable.
        spatial_subset_path = os.path.join(wd, 'spatial_subset.nc')
        _write_spatial_subset_(rd_src, rd_dst, spatial_subset_path)
    # Only split grids if a spatial subset is not requested.
    else:
        # Update the paths to use for the grid.
        paths = {'wd': wd}

    # Arguments to ESMF regridding.
    esmf_kwargs = {'regrid_method': esmf_regrid_method,
                   'ignore_degenerate': ignore_degenerate}

    # Create the chunked regridding object. This is used for both chunked regridding and a regrid with a spatial subset.
    gs = GridChunker(rd_src, rd_dst, nchunks_dst=nchunks_dst, src_grid_resolution=src_resolution, paths=paths,
                     dst_grid_resolution=dst_resolution, buffer_value=buffer_distance, redistribute=True,
                     genweights=genweights, esmf_kwargs=esmf_kwargs, use_spatial_decomp='auto', eager=eager)

    # Write subsets and generate weights if requested in the grid splitter.
    # TODO: Need a weight only option. If chunks are written, then weights are written...
    if not spatial_subset and nchunks_dst is not None:
        gs.write_chunks()
    else:
        if spatial_subset:
            source = spatial_subset_path
        if genweights:
            gs.write_esmf_weights(source, destination, weight)

    # Create the global weight file. This does not apply to spatial subsets because there will always be one weight
    # file.
    if merge and not spatial_subset:
        # Weight file merge only works in serial.
        exc = None
        with ocgis.vm.scoped('weight file merge', [0]):
            if not ocgis.vm.is_null:
                gs.create_merged_weight_file(weight)
        excs = ocgis.vm.gather(exc)
        excs = ocgis.vm.bcast(excs)
        for exc in excs:
            if exc is not None:
                raise exc

        ocgis.vm.barrier()

    # Remove the working directory unless the persist flag is provided.
    if not persist:
        if ocgis.vm.rank == 0:
            shutil.rmtree(wd)
        ocgis.vm.barrier()

    return 0


def _create_request_dataset_(path, esmf_type):
    edmap = {'GRIDSPEC': DriverKey.NETCDF_CF,
             'UGRID': DriverKey.NETCDF_UGRID,
             'SCRIP': DriverKey.NETCDF_SCRIP}
    odriver = edmap[esmf_type]
    return RequestDataset(uri=path, driver=odriver, grid_abstraction='point')


def _is_subdir_(path, potential_subpath):
    # https://stackoverflow.com/questions/3812849/how-to-check-whether-a-directory-is-a-sub-directory-of-another-directory#18115684
    path = os.path.realpath(path)
    potential_subpath = os.path.realpath(potential_subpath)
    relative = os.path.relpath(path, potential_subpath)
    return not relative.startswith(os.pardir + os.sep)


def _write_spatial_subset_(rd_src, rd_dst, spatial_subset_path):
    src_field = rd_src.create_field()
    dst_field = rd_dst.create_field()
    sso = SpatialSubsetOperation(src_field)

    with grid_abstraction_scope(dst_field.grid, Topology.POLYGON):
        dst_field_extent = dst_field.grid.extent_global

    subset_geom = GeometryVariable.from_shapely(box(*dst_field_extent), crs=dst_field.crs, is_bbox=True)
    buffer_value = GridChunkerConstants.BUFFER_RESOLUTION_MODIFIER * src_field.grid.resolution_max
    sub_src = sso.get_spatial_subset('intersects', subset_geom, buffer_value=buffer_value, optimized_bbox_subset=True)

    # Try to reduce the coordinate indexing for unstructured grids.
    try:
        reduced = sub_src.grid.reduce_global()
    except AttributeError:
        pass
    else:
        sub_src = reduced.parent

    sub_src.write(spatial_subset_path)


if __name__ == '__main__':
    ocli()
