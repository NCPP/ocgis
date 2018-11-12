from ocgis.base import get_dimension_names

from ocgis.constants import KeywordArgument, VariableName

from ocgis import DimensionMap
from ocgis.driver.nc import DriverNetcdfCF
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

        if sobj.has_mask:
            mask_variable = sobj.mask_variable
        else:
            if create:
                mask_value = np.zeros(sobj.shape, dtype=bool)
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


def create_dimension_map(meta, driver):
    # tdk: DOC
    # Check if this is a class or an instance. If it is a class, convert to instance for dimension map
    # creation.
    if isinstance(driver, type):
        driver = driver()
    dimmap = DimensionMap.from_metadata(driver, meta)
    return dimmap


def create_metadata_from_xarray(ds):
    # tdk: DOC
    xmeta = Dict()
    for dimname, dimsize in ds.dims.items():
        xmeta.dimensions[dimname] = {'name': dimname, 'size': dimsize}
    for varname, var in ds.variables.items():
        xmeta.variables[varname] = {'name': varname, 'dimensions': var.dims, 'attrs': var.attrs}
    xmeta = dict(xmeta)
    return xmeta