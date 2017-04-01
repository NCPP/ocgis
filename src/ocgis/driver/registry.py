from ocgis.base import AbstractOcgisObject
from ocgis.constants import DriverKeys
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


def get_driver_class(key_or_class, default=None):
    ret = None

    if key_or_class == AbstractDriver or key_or_class == DriverKeys.BASE:
        ret = AbstractDriver
    else:
        if default is not None:
            default = get_driver_class(default)

        for driver in driver_registry.drivers:
            if key_or_class == driver:
                ret = key_or_class
            else:
                try:
                    if driver.key == key_or_class:
                        ret = driver
                except AttributeError:
                    continue

    if ret is None:
        if default is None:
            raise ValueError('Driver "{}" not found. Is it append to "driver.registry"?'.format(key_or_class))
        else:
            ret = default

    return ret
