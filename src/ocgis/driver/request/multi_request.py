import six

from ocgis.driver.request.base import AbstractRequestObject
from ocgis.util.helpers import get_iter


class MultiRequestDataset(AbstractRequestObject):
    """
    Acts like a single request dataset. No dimension checking is done when pulling variables into a single field. The
    first field in `request_datasets` is used as the archetype destination for the data variables found in other request
    datasets.

    :param sequence request_datasets: A sequence of :class:`ocgis.RequestDataset` objects. Must be sliceable.
    """

    def __init__(self, request_datasets):
        self.request_datasets = request_datasets

    @property
    def crs(self):
        return get_request_dataset_attribute(self, 'crs')

    @property
    def dimension_map(self):
        return get_request_dataset_attribute(self, 'dimension_map')

    @property
    def driver(self):
        return get_request_dataset_attribute(self, 'driver')

    @property
    def field_name(self):
        return get_request_dataset_attribute(self, 'field_name')

    @property
    def has_data_variables(self):
        return get_request_dataset_attribute(self, 'has_data_variables')

    @property
    def metadata(self):
        build = True
        for rd in self.request_datasets:
            if build:
                ret = rd.metadata.copy()
                build = False
            else:
                for variable_name in get_iter(rd.variable):
                    ret['variables'][variable_name] = rd.metadata['variables'][variable_name]
        return ret

    @property
    def regrid_destination(self):
        return get_request_dataset_attribute(self, 'regrid_destination')

    @property
    def uid(self):
        return get_request_dataset_attribute(self, 'uid')

    @uid.setter
    def uid(self, value):
        set_request_dataset_attribute(self, 'uid', value)

    @property
    def units(self):
        return get_request_dataset_iterable_attribute(self, 'units')

    @property
    def uri(self):
        return get_request_dataset_attribute(self, 'uri')

    @property
    def variable(self):
        return get_request_dataset_iterable_attribute(self, 'variable')

    def get(self, **kwargs):
        build = True
        for rd in self.request_datasets:
            current = get_field(rd, **kwargs)
            if build:
                target = current
                build = False
            else:
                for variable in iter_variables_to_add(current):
                    variable = variable.extract()
                    target.add_variable(variable, is_data=True)
        return target

    def _get_meta_rows_(self):
        return get_request_dataset_attribute(self, '_get_meta_rows_')()


def get_field(target, **kwargs):
    return target.get(**kwargs)


def get_request_dataset_attribute(obj, attr):
    target = obj.request_datasets[0]
    return getattr(target, attr)


def get_request_dataset_iterable_attribute(obj, attr):
    nested = [getattr(target, attr) for target in obj.request_datasets]
    flattened = []
    for n in get_iter(nested):
        if isinstance(n, six.string_types) or n is None:
            flattened.append(n)
        else:
            flattened += list(n)
    return tuple(flattened)


def set_request_dataset_attribute(obj, attr, value):
    target = obj.request_datasets[0]
    setattr(target, attr, value)


def iter_variables_to_add(target):
    for yld in target.data_variables:
        yield yld
