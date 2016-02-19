from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from warnings import warn

import numpy as np


class AbstractMetadata(OrderedDict):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lines(self):
        pass

    @abstractmethod
    def _parse_(self):
        pass


class NcMetadata(AbstractMetadata):
    """
    :param rootgrp: An open NetCDF4 dataset object.
    :type rootgrp: :class:`netCDF4.Dataset`
    """

    def __init__(self, rootgrp=None):
        super(NcMetadata, self).__init__()

        if rootgrp is not None:
            try:
                self._parse_(rootgrp)
            # likely raised by an initialization following a deepcopy
            except AttributeError:
                super(NcMetadata, self).__init__(rootgrp)

    def get_lines(self):
        lines = ['dimensions:']
        template = '    {0} = {1} ;{2}'
        for key, value in self['dimensions'].iteritems():
            if value['isunlimited']:
                one = 'ISUNLIMITED'
                two = ' // {0} currently'.format(value['len'])
            else:
                one = value['len']
                two = ''
            lines.append(template.format(key, one, two))

        lines.append('')
        lines.append('variables:')
        var_template = '    {0} {1}({2}) ;'
        attr_template = '      {0}:{1} = "{2}" ;'
        for key, value in self['variables'].iteritems():
            dims = [str(d) for d in value['dimensions']]
            dims = ', '.join(dims)
            lines.append(var_template.format(value['dtype'], key, dims))
            for key2, value2 in value['attrs'].iteritems():
                lines.append(attr_template.format(key, key2, value2))

        lines.append('')
        lines.append('// global attributes:')
        template = '    :{0} = {1} ;'
        for key, value in self['dataset'].iteritems():
            try:
                lines.append(template.format(key, value))
            except UnicodeEncodeError:
                # for a unicode string, if "\u" is in the string and an inappropriate unicode character is used, then
                # template formatting will break.
                msg = 'Unable to encode attribute "{0}". Skipping printing of attribute value.'.format(key)
                warn(msg)

        return lines

    def _parse_(self, rootgrp):
        # get global metadata
        dataset = OrderedDict()
        for attr in rootgrp.ncattrs():
            dataset.update({attr: getattr(rootgrp, attr)})
        self.update({'dataset': dataset})

        # get file format
        self.update({'file_format': rootgrp.file_format})

        # get variables
        variables = OrderedDict()
        for key, value in rootgrp.variables.iteritems():
            subvar = OrderedDict()
            for attr in value.ncattrs():
                if attr.startswith('_'):
                    continue
                subvar.update({attr: getattr(value, attr)})

            # Remove scale factors and offsets from the metadata.
            if 'scale_factor' in subvar:
                dtype_packed = value[0].dtype
                fill_value_packed = np.ma.array([], dtype=dtype_packed).fill_value
            else:
                dtype_packed = None
                fill_value_packed = None

            # make two attempts at missing value attributes otherwise assume the default from a numpy masked array
            try:
                fill_value = value.fill_value
            except AttributeError:
                try:
                    fill_value = value.missing_value
                except AttributeError:
                    fill_value = np.ma.array([], dtype=value.dtype).fill_value

            variables.update({key: {'dimensions': value.dimensions,
                                    'attrs': subvar,
                                    'dtype': str(value.dtype),
                                    'name': value._name,
                                    'fill_value': fill_value,
                                    'dtype_packed': dtype_packed,
                                    'fill_value_packed': fill_value_packed}})
        self.update({'variables': variables})

        # get dimensions
        dimensions = OrderedDict()
        for key, value in rootgrp.dimensions.iteritems():
            subdim = {key: {'len': len(value), 'isunlimited': value.isunlimited()}}
            dimensions.update(subdim)
        self.update({'dimensions': dimensions})
