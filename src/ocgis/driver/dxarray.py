from ocgis.base import get_dimension_names

from ocgis.constants import KeywordArgument, VariableName
from ocgis.driver.base import AbstractUnstructuredDriver

from ocgis.driver.nc import DriverNetcdfCF
from ocgis.driver.nc_esmf_unstruct import DriverESMFUnstruct
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.util.addict import Dict
import xarray as xr
import numpy as np


class DriverXarray(DriverNetcdfCF):
    key = 'xarray'
    extensions = []
    output_formats = []

    @staticmethod
    def create_metadata(field):
        return create_metadata_from_xarray(field._storage)

    @staticmethod
    def create_varlike(value, **kwargs):
        # tdk: DOC

        kwargs = kwargs.copy()

        # Allow for dimension objects to come in as "dims".
        if 'dims' in kwargs:
            kwargs['dims'] = get_dimension_names(kwargs['dims'])

        return xr.DataArray(value, **kwargs)

    @staticmethod
    def get_bounds(varlike, container):
        # tdk: DOC
        ret = None
        if hasattr(varlike, 'bounds'):
            try:
                ret = container[varlike.bounds]
            except KeyError:
                # Occurs when the attribute is present but the data array is not in the dataset.
                assert varlike.bounds not in container
        return ret

    @staticmethod
    def get_or_create_spatial_mask(*args, **kwargs):
        # tdk: DOC
        args = list(args)
        sobj = args[0]
        create = kwargs.get(KeywordArgument.CREATE, False)
        initial_value = kwargs.get(KeywordArgument.INIT_VALUE, None)

        if sobj.has_mask:
            mask_variable = sobj.mask_variable
        else:
            if create:
                if initial_value is None:
                    mask_value = np.zeros(sobj.shape, dtype=bool)
                else:
                    mask_value = initial_value
                mask_variable = xr.DataArray(mask_value,
                                             name=VariableName.SPATIAL_MASK,
                                             dims=get_dimension_names(sobj.dimensions),
                                             attrs={'ocgis_role': 'spatial_mask'})
                sobj.set_mask(mask_variable)
            else:
                mask_variable = None

        if mask_variable is None:
            ret = None
        else:
            ret = mask_variable.values

        return ret

    @staticmethod
    def get_value(target, **kwargs):
        # tdk: DOC
        return target.values

    def get_variable_value(self, *args, **kwargs):
        raise NotImplementedError

    def _get_metadata_main_(self):
        raise NotImplementedError

    def _init_variable_from_source_main_(self, *args, **kwargs):
        raise NotImplementedError

    def _write_variable_collection_main_(cls, *args, **kwargs):
        raise NotImplementedError


class DriverXarrayUGRID(DriverXarray, DriverNetcdfUGRID):
    key = "xarray-ugrid"


class DriverXarrayESMFUnstruct(DriverXarray, DriverESMFUnstruct):
    key = "xarray-esmf-unstruct"


def create_dask_chunk_defn(group_metadata, chunked_dims, size):
    # tdk:doc
    from ocgis.vmachine.mpi import OcgDist
    from ocgis import Dimension
    assert len(chunked_dims) > 0
    dimmeta = group_metadata['dimensions']
    dist = OcgDist(size=size)
    for k, v in dimmeta.items():
        if k in chunked_dims:
            dim = Dimension(name=k, size=v['size'], dist=True)
            dist.add_dimension(dim)
    dist.update_dimension_bounds()
    ret = {}
    for d in chunked_dims:
        bl = dist.get_dimension(d, rank=0).bounds_local
        assert (len(bl) > 0)
        ret[d] = bl[1] - bl[0]
    return ret


def create_metadata_from_xarray(ds):
    """
    Create a standard metadata dictionary from an ``xarray`` ``Dataset``.

    :param ds: Source dataset
    :type ds: :class:`~xarray.core.dataset.Dataset`
    :return: Standard ``ocgis`` metadata dictionary
    :rtype: dict
    """
    xmeta = Dict()
    for dimname, dimsize in ds.dims.items():
        xmeta.dimensions[dimname] = {'name': dimname, 'size': dimsize}
    for varname, var in ds.variables.items():
        xmeta.variables[varname] = {'name': varname, 'dimensions': var.dims, 'attrs': var.attrs}
    xmeta = xmeta.to_dict()
    return xmeta