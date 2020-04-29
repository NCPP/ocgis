#!/usr/bin/env python
import datetime
import logging
import os
import shutil
import tempfile

import click
import netCDF4 as nc
from shapely.geometry import box

import ocgis
from ocgis import RequestDataset, GeometryVariable, constants
from ocgis.base import grid_abstraction_scope, raise_if_empty
from ocgis.constants import DriverKey, Topology, GridChunkerConstants, DecompositionType
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.messages import M5
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.spatial.spatial_subset import SpatialSubsetOperation
from ocgis.util.logging_ocgis import ocgis_lh

CRWG_LOG = "chunked-rwg"


def handle_weight_file_check(path):
    if path is not None and os.path.exists(path):
        exc = IOError("Weight file must be removed before writing a new new one: {}".format(path))
        try:
            raise exc
        finally:
            ocgis.vm.abort(exc=exc)


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
              help='(default=GRIDSPEC) ESMF source grid type. Supports GRIDSPEC, UGRID, ESMFMESH, and SCRIP.')
@click.option('--esmf_dst_type', type=str, nargs=1, default='GRIDSPEC',
              help='(default=GRIDSPEC) ESMF destination grid type. Supports GRIDSPEC, UGRID, ESMFMESH, and SCRIP.')
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
@click.option('--data_variables', default=None, type=str,
              help='List of comma-separated data variable names to overload auto-discovery.')
@click.option('--spatial_subset_path', default=None, type=click.Path(dir_okay=False),
              help='Optional path to the output spatial subset file. Only applicable when using --spatial_subset.')
@click.option('--verbose/--not_verbose', default=False, help='If True, log to standard out using verbosity level.')
@click.option('--loglvl', default="INFO", help='Verbosity level for standard out logging. Default is '
              '"INFO". See Python logging level docs for additional values: https://docs.python.org/3/howto/logging.html')
@click.option('--weightfilemode', default="BASIC", help=M5)
@click.option('--esmf_global_index/--not_esmf_global_index', default=True)
def chunked_rwg(source, destination, weight, nchunks_dst, merge, esmf_src_type, esmf_dst_type, genweights,
                esmf_regrid_method, spatial_subset, src_resolution, dst_resolution, buffer_distance, wd, persist,
                eager, ignore_degenerate, data_variables, spatial_subset_path, verbose, loglvl, weightfilemode,
                esmf_global_index):
    #tdk:doc: esmf_global_index

    # Used for creating the history string.
    the_locals = locals()

    if verbose:
        ocgis_lh.configure(to_stream=True, level=getattr(logging, loglvl))
    ocgis_lh(msg="Starting Chunked Regrid Weight Generation", level=logging.INFO, logger=CRWG_LOG)

    if not ocgis.env.USE_NETCDF4_MPI:
        msg = ('env.USE_NETCDF4_MPI is False. Considerable performance gains are possible if this is True. Is '
               'netCDF4-python built with parallel support?')
        ocgis_lh(msg, level=logging.WARN, logger=CRWG_LOG, force=True)

    if data_variables is not None:
        data_variables = data_variables.split(',')

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
    should_create_wd = (nchunks_dst is None or not all([ii == 1 for ii in nchunks_dst])) or spatial_subset
    if should_create_wd:
        if wd is None:
            if ocgis.vm.rank == 0:
                wd = tempfile.mkdtemp(prefix='ocgis_chunked_rwg_')
            wd = ocgis.vm.bcast(wd)
        else:
            exc = None
            if ocgis.vm.rank == 0:
                # The working directory must not exist to proceed.
                if nchunks_dst is not None:
                    if os.path.exists(wd):
                        exc = ValueError("Working directory {} must not exist.".format(wd))
                    else:
                        # Make the working directory nesting as needed.
                        os.makedirs(wd)
            exc = ocgis.vm.bcast(exc)
            if exc is not None:
                raise exc

        if merge and not spatial_subset or (spatial_subset and genweights):
            if _is_subdir_(wd, weight):
                raise ValueError(
                    'Merge weight file path must not in the working directory. It may get unintentionally deleted with the --no_persist flag.')

    # Create the source and destination request datasets.
    rd_src = _create_request_dataset_(source, esmf_src_type, data_variables=data_variables)
    rd_dst = _create_request_dataset_(destination, esmf_dst_type)

    # Execute a spatial subset if requested.
    paths = None
    if spatial_subset:
        if spatial_subset_path is None:
            spatial_subset_path = os.path.join(wd, 'spatial_subset.nc')
        msg = "Executing spatial subset. Output path is: {}".format(spatial_subset_path)
        ocgis_lh(msg=msg, level=logging.INFO, logger=CRWG_LOG)
        _write_spatial_subset_(rd_src, rd_dst, spatial_subset_path, src_resmax=src_resolution,
                               esmf_global_index=esmf_global_index)
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
    if not spatial_subset and nchunks_dst is not None and not gs.is_one_chunk:
        msg = "Starting main chunking loop..."
        ocgis_lh(msg=msg, level=logging.INFO, logger=CRWG_LOG)
        gs.write_chunks()
    else:
        if spatial_subset:
            source = spatial_subset_path
        if genweights:
            msg = "Writing ESMF weights..."
            ocgis_lh(msg=msg, level=logging.INFO, logger=CRWG_LOG)
            handle_weight_file_check(weight)
            gs.write_esmf_weights(source, destination, weight, filemode=weightfilemode)

    # Create the global weight file. This does not apply to spatial subsets because there will always be one weight
    # file.
    if merge and not spatial_subset and not gs.is_one_chunk:
        # Weight file merge only works in serial.
        exc = None
        with ocgis.vm.scoped('weight file merge', [0]):
            if not ocgis.vm.is_null:
                msg = "Merging chunked weight files to global file. Output global weight file is: {}".format(weight)
                ocgis_lh(msg=msg, level=logging.INFO, logger=CRWG_LOG)
                handle_weight_file_check(weight)
                gs.create_merged_weight_file(weight)
        excs = ocgis.vm.gather(exc)
        excs = ocgis.vm.bcast(excs)
        for exc in excs:
            if exc is not None:
                raise exc

        ocgis.vm.barrier()

    # Append the history string if there is an output weight file.
    if weight and ocgis.vm.rank == 0:
        if os.path.exists(weight):
            # Add some additional stuff for record keeping
            import getpass
            import socket
            import datetime

            with nc.Dataset(weight, 'a') as ds:
                ds.setncattr('created_by_user', getpass.getuser())
                ds.setncattr('created_on_hostname', socket.getfqdn())
                ds.setncattr('history', create_history_string(the_locals))
    ocgis.vm.barrier()

    # Remove the working directory unless the persist flag is provided.
    if not persist:
        if ocgis.vm.rank == 0:
            msg = "Removing working directory since persist is False."
            ocgis_lh(msg=msg, level=logging.INFO, logger=CRWG_LOG)
            shutil.rmtree(wd)
        ocgis.vm.barrier()

    ocgis_lh(msg="Success!", level=logging.INFO, logger=CRWG_LOG)
    return 0


def create_history_string(the_locals):
    history_parms = {}
    for k, v in the_locals.items():
        if v is not None and k != 'history_parms':
            if type(v) == bool:
                if not v:
                    history_parms['--no_' + k] = v
            else:
                history_parms['--' + k] = v
    try:
        import ESMF
        ever = ESMF.__version__
    except ImportError:
        ever = None
    history = "{}: Created by ocgis (v{}) and ESMF (v{}) with CLI command: ocli chunked-rwg".format(
        datetime.datetime.now(), ocgis.__version__, ever)
    for k, v in history_parms.items():
        history += " {} {}".format(k, v)
    return history


@ocli.command(help='Apply weights in chunked files with an option to insert the global data.', name='chunked-smm')
@click.option('--wd', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Optional working directory containing destination chunk files. If empty, the current working "
                   "directory is used.")
@click.option('--index_path', required=False, type=click.Path(exists=True, dir_okay=False),
              help='Path grid chunker index file. If not provided, it will assume the default name in the working '
                   'directory.')
@click.option('--insert_weighted/--no_insert_weighted', default=False, required=False,
              help='If --insert_weighted, insert the weighted data back into the global destination file.')
@click.option('-d', '--destination', required=False, type=click.Path(exists=True, dir_okay=False),
              help='Path to the destination grid NetCDF file. Needed if using --insert_weighted.')
@click.option('--data_variables', default='auto', type=str,
              help='List of comma-separated data variable names to overload auto-discovery.')
def chunked_smm(wd, index_path, insert_weighted, destination, data_variables):
    if wd is None:
        wd = os.getcwd()

    if data_variables != 'auto':
        data_variables = data_variables.split(',')

    if index_path is None:
        index_path = os.path.join(wd, constants.GridChunkerConstants.DEFAULT_PATHS['index_file'])
        ocgis.vm.barrier()
        assert os.path.exists(index_path)

    if insert_weighted:
        if destination is None:
            raise ValueError('If --insert_weighted, then "destination" must be provided.')

    # ------------------------------------------------------------------------------------------------------------------

    GridChunker.smm(index_path, wd, data_variables=data_variables)
    if insert_weighted:
        with ocgis.vm.scoped_barrier(first=True, last=True):
            with ocgis.vm.scoped('insert weighted', [0]):
                if not ocgis.vm.is_null:
                    GridChunker.insert_weighted(index_path, wd, destination, data_variables=data_variables)


def _create_request_dataset_(path, esmf_type, data_variables='auto'):
    edmap = {'GRIDSPEC': DriverKey.NETCDF_CF,
             'UGRID': DriverKey.NETCDF_UGRID,
             'SCRIP': DriverKey.NETCDF_SCRIP,
             'ESMFMESH': DriverKey.NETCDF_ESMF_UNSTRUCT}
    odriver = edmap[esmf_type]
    if data_variables == 'auto':
        v = None
    else:
        v = data_variables
    return RequestDataset(uri=path, driver=odriver, grid_abstraction='point', variable=v,
                          decomp_type=DecompositionType.ESMF)


def _is_subdir_(path, potential_subpath):
    # https://stackoverflow.com/questions/3812849/how-to-check-whether-a-directory-is-a-sub-directory-of-another-directory#18115684
    path = os.path.realpath(path)
    potential_subpath = os.path.realpath(potential_subpath)
    relative = os.path.relpath(path, potential_subpath)
    return not relative.startswith(os.pardir + os.sep)


def _write_spatial_subset_(rd_src, rd_dst, spatial_subset_path, src_resmax=None, esmf_global_index=False):
    src_field = rd_src.create_field()
    dst_field = rd_dst.create_field()
    sso = SpatialSubsetOperation(src_field, add_esmf_index=esmf_global_index)

    with grid_abstraction_scope(dst_field.grid, Topology.POLYGON):
        dst_field_extent = dst_field.grid.extent_global

    subset_geom = GeometryVariable.from_shapely(box(*dst_field_extent), crs=dst_field.crs, is_bbox=True)
    if src_resmax is None:
        src_resmax = src_field.grid.resolution_max
    buffer_value = GridChunkerConstants.BUFFER_RESOLUTION_MODIFIER * src_resmax
    sub_src = sso.get_spatial_subset('intersects', subset_geom, buffer_value=buffer_value, optimized_bbox_subset=True)
    # No empty spatial subsets allowed through CLI. There will be nothing for ESMF to do.
    raise_if_empty(sub_src, check_current=True)

    # Try to reduce the coordinate indexing for unstructured grids.
    with ocgis.vm.scoped_by_emptyable('subset reduce/write', sub_src):
        if not ocgis.vm.is_null:
            # Attempt to reindex the subset.
            try:
                reduced = sub_src.grid.reduce_global()
            except AttributeError:
                pass
            except ValueError:
                if sub_src.driver.__class__ == DriverNetcdfUGRID:
                    raise
            else:
                sub_src = reduced.parent

            # Write the subset to file.
            sub_src.write(spatial_subset_path)


if __name__ == '__main__':
    ocli()
