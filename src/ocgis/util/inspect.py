# noinspection PyPep8Naming
import netCDF4 as nc
from collections import OrderedDict
import re
from warnings import warn

from ocgis.exc import NoDimensionedVariablesFound
from ocgis.api.request.base import RequestDataset


class Inspect(object):
    """
    Inspect a dataset printing information to stdout. If a variable target is available, a field object will be
    constructed from the request dataset and summary information for the variable printed in addition to the metadata
    dump.

    >>> from ocgis import Inspect
    >>> uri = '/path/to/dataset.nc'
    >>> variable = 'tas'
    >>> print Inspect(uri=uri, variable=variable)

    :param uri: The URI of the target dataset.
    :type uri: string or sequence
    :param str variable: The target variable to inspect.
    :param request_dataset: The request dataset object to inspect.
    :type request_dataset: :class:`~ocgis.api.request.base.RequestDataset`
    :raises: ValueError
    """

    newline = '\n'

    def __init__(self, uri=None, variable=None, request_dataset=None):
        if uri is None and request_dataset is None:
            msg = 'A URI or request dataset is required.'
            raise ValueError(msg)

        self._variable = variable

        self.uri = uri
        if request_dataset is None:
            request_dataset = RequestDataset(uri=uri, variable=variable)
        self.request_dataset = request_dataset

    def __str__(self):
        lines = self.get_report_possible()
        return self.newline.join(lines)

    @property
    def variable(self):
        if self._variable is None:
            try:
                ret = self.request_dataset.variable
            except NoDimensionedVariablesFound:
                ret = None
        else:
            ret = self._variable
        return ret

    def get_field_report(self):

        def _getattr_(obj, target):
            try:
                ret = getattr(obj, target)
            except AttributeError:
                if obj is None:
                    ret = obj
                else:
                    raise
            return ret

        field = self.request_dataset.get()
        m = OrderedDict([['=== Temporal =============', 'temporal'],
                         ['=== Spatial ==============', 'spatial'],
                         ['=== Level ================', 'level']])
        lines = []
        for k, v in m.iteritems():
            sub = [k, '']
            dim = _getattr_(field, v)
            if dim is None:
                sub.append('No {0} dimension.'.format(v))
            else:
                sub += dim.get_report()
            sub.append('')
            lines += sub

        return lines

    def get_header(self):
        lines = ['URI = {0}'.format(self.request_dataset.uri),
                 'VARIABLE = {0}'.format(self.variable)]
        return lines

    def get_report(self):
        lines = self.get_header()
        lines.append('')
        lines += self.get_field_report()
        self._append_dump_report_(lines)
        return lines

    def get_report_no_field(self):
        lines = self.get_header()
        lines.append('')
        self._append_dump_report_(lines)
        return lines

    def get_report_possible(self):
        if self.variable is None:
            lines = self.get_report_no_field()
        else:
            lines = self.get_report()
        return lines

    def _append_dump_report_(self, target):
        """
        :type target: sequence[str, ...]
        """

        target.append('=== Metadata Dump ========')
        target.append('')
        target += self.request_dataset.driver.get_dump_report()

    def _as_dct_(self):
        ret = self.request_dataset.source_metadata.copy()
        # without a target variable, attempt to set start and end dates.
        if self.variable is None:
            ds = nc.Dataset(self.uri, 'r')
            # noinspection PyBroadException
            try:
                time = ds.variables['time']
                time_bounds = [time[0], time[-1]]
                time_bounds = nc.num2date(time_bounds, time.units, calendar=time.calendar)
                derived = {'Start Date': str(time_bounds[0]), 'End Date': str(time_bounds[1])}
            except:
                warn('Time variable not found or improperly attributed. Setting "derived" key to None.')
                derived = None
            finally:
                ds.close()
        # we can get derived values
        else:
            derived = OrderedDict()
            to_add = self.get_field_report()
            for row in to_add:
                try:
                    key, value = re.split(' = ', row, maxsplit=1)
                # here to catch oddities of the returns
                except ValueError:
                    continue
                key = key.strip()
                derived.update({key: value})
        ret.update({'derived': derived})
        return ret
