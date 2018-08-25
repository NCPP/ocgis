from ocgis.driver.base import AbstractDriver


class DriverXarray(AbstractDriver):
    key = 'xarray'
    extensions = None
    output_formats = None

    def get_variable_value(self, variable):
        raise NotImplementedError

    def _get_metadata_main_(self):
        raise NotImplementedError

    def _init_variable_from_source_main_(self, *args, **kwargs):
        raise NotImplementedError

    def _write_variable_collection_main_(cls, *args, **kwargs):
        raise NotImplementedError