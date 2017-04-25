from ocgis.base import AbstractOcgisObject
from ocgis.constants import DriverKey
from .base import AbstractDriver
from .csv_ import DriverCSV
from .nc import DriverNetcdf, DriverNetcdfCF
from .vector import DriverVector


class DriverRegistry(AbstractOcgisObject):
    def __init__(self):
        self.drivers = []


driver_registry = DriverRegistry()
driver_registry.drivers.append(DriverCSV)
driver_registry.drivers.append(DriverNetcdf)
driver_registry.drivers.append(DriverNetcdfCF)
driver_registry.drivers.append(DriverVector)


def get_driver_class(key_class_or_instance, default=None):
    ret = None

    # Allow driver instances to pass through.
    if isinstance(key_class_or_instance, AbstractDriver):
        ret = key_class_or_instance
    elif key_class_or_instance == AbstractDriver or key_class_or_instance == DriverKey.BASE:
        ret = AbstractDriver
    else:
        if default is not None:
            default = get_driver_class(default)

        for driver in driver_registry.drivers:
            if key_class_or_instance == driver:
                ret = key_class_or_instance
            else:
                try:
                    if driver.key == key_class_or_instance:
                        ret = driver
                except AttributeError:
                    continue

    if ret is None:
        if default is None:
            raise ValueError(
                'Driver "{}" not found. Is it appended to "driver.registry"?'.format(key_class_or_instance))
        else:
            ret = default

    return ret
