from ocgis import DimensionMap
from ocgis.driver.nc import DriverNetcdfCF
from ocgis.util.addict import Dict


class DriverXarray(DriverNetcdfCF):
    key = 'xarray'
    extensions = []
    output_formats = []

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