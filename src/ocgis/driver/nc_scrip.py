import numpy as np

from ocgis import DimensionMap, env, constants, vm
from ocgis.base import raise_if_empty
from ocgis.constants import DriverKey, DMK, Topology
from ocgis.driver.base import AbstractUnstructuredDriver
from ocgis.driver.nc import DriverNetcdf
from ocgis.util.helpers import create_unique_global_array
from ocgis.vmachine.mpi import hgather


class DriverNetcdfSCRIP(AbstractUnstructuredDriver, DriverNetcdf):
    """
    Driver for the SCRIP NetCDF structured and unstructured grid format. SCRIP is a legacy format that is the primary
    precursor to NetCDF-CF convention. By default, SCRIP grids are treated as unstructured data creating an unstructured
    grid.
    """

    _esmf_fileformat = 'SCRIP'
    key = DriverKey.NETCDF_SCRIP
    _default_crs = env.DEFAULT_COORDSYS

    @staticmethod
    def array_resolution(value, axis):
        """See :meth:`ocgis.driver.base.AbstractDriver.array_resolution`"""
        if value.size == 1:
            return 0.0
        else:
            resolution_limit = constants.RESOLUTION_LIMIT
            value = np.sort(np.unique(np.abs(value)))
            value = value[0:resolution_limit]
            value = np.diff(value)
            ret = np.mean(value)
            return ret

    def create_dimension_map(self, group_metadata, **kwargs):
        ret = DimensionMap()
        ret.set_driver(self)

        topo = ret.get_topology(Topology.POINT, create=True)
        topo.set_variable(DMK.X, 'grid_center_lon', dimension='grid_size')
        topo.set_variable(DMK.Y, 'grid_center_lat', dimension='grid_size')

        if 'grid_corner_lon' in group_metadata['variables']:
            topo = ret.get_topology(Topology.POLYGON, create=True)
            topo.set_variable(DMK.X, 'grid_corner_lon', dimension='grid_size')
            topo.set_variable(DMK.Y, 'grid_corner_lat', dimension='grid_size')

        if 'grid_imask' in group_metadata['variables']:
            # Use the intrinsic SCRIP default attributes associated with the variable.
            ret.set_spatial_mask('grid_imask', default_attrs={})

        # The isomorphic property covers all possible mesh topologies.
        ret.set_property(DMK.IS_ISOMORPHIC, True)

        return ret

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata):
        return 'grid_size'

    @staticmethod
    def get_or_create_spatial_mask(*args, **kwargs):
        """
        Get or create the SCRIP spatial mask. Arguments and keyword arguments in signature are for driver
        compatibility only.
        """
        sobj = args[0]
        raise_if_empty(sobj)

        ret = None
        if sobj.has_mask:
            maskvar = sobj.parent['grid_imask']
            if not maskvar.has_allocated_value:
                v = maskvar.v()
                ret = v == 0
                maskvar.set_mask(ret)
            else:
                ret = maskvar.m()
        return ret

    @staticmethod
    def validate_spatial_mask(mask_variable):
        if mask_variable.name != 'grid_imask':
            msg = 'For SCRIP data, the mask variable must be named "grid_imask".'
            raise ValueError(msg)

    @classmethod
    def _get_field_write_target_(cls, field):
        # Unstructured SCRIP has a value of 1 for the grid dimensions by default. Just leave it alone.
        if field.dimensions['grid_rank'].size > 1:
            # Update the grid size based on unique x/y values. In SCRIP, the coordinate values are duplicated in the
            # coordinate vector.
            ux = field.grid.x.shape[0]
            uy = field.grid.y.shape[0]
            field['grid_dims'].get_value()[:] = ux, uy
        return field

    @staticmethod
    def _gc_iter_dst_grid_slices_(grid_chunker):
        # TODO: This method uses some global gathers which is not ideal.
        # Destination splitting works off center coordinates only.
        pgc = grid_chunker.dst_grid.abstractions_available['point']

        # Use the unique center values to break the grid into pieces. This ensures that nearby grid cell are close
        # spatially. If we just break the grid into pieces w/out using unique values, the points may be scattered which
        # does not optimize the spatial coverage of the source grid.
        center_lat = pgc.y.get_value()

        # ucenter_lat = np.unique(center_lat)
        ucenter_lat = create_unique_global_array(center_lat)

        ucenter_lat = vm.gather(ucenter_lat)
        if vm.rank == 0:
            ucenter_lat = hgather(ucenter_lat)
            ucenter_lat.sort()
            ucenter_splits = np.array_split(ucenter_lat, grid_chunker.nchunks_dst[0])
        else:
            ucenter_splits = [None] * grid_chunker.nchunks_dst[0]

        for ucenter_split in ucenter_splits:
            ucenter_split = vm.bcast(ucenter_split)
            select = np.zeros_like(center_lat, dtype=bool)
            for v in ucenter_split.flat:
                select = np.logical_or(select, center_lat == v)
            yield select

    @staticmethod
    def _gc_nchunks_dst_(grid_chunker):
        g = grid_chunker.dst_grid
        size = g.y.size
        if size < 100:
            ret = int(size / 2)
        else:
            raise NotImplementedError
        return ret
