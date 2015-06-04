from ocgis import constants
import abc
import datetime

import ocgis
from ocgis.api.parms.base import AbstractParameter
from ocgis.conv.base import AbstractConverter
from ocgis.exc import DefinitionValidationError
from ocgis.util.justify import justify_row


HEADERS = {
    'ugid': 'User geometry identifier pulled from a provided set of selection geometries. Reduces to "1" for the case of no provided geometry.',
    'gid': 'Geometry identifier assigned by OpenClimateGIS to a dataset geometry. In the case of "aggregate=True" this is equivalent to "UGID".',
    'tid': 'Unique time identifier.',
    'vid': 'Unique variable identifier.',
    'lid': 'Level identifier unique within a variable.',
    'name': 'Name of the requested variable.',
    'calc_alias': 'User-supplied name for a calculation.',
    'calc_key': 'The unique key name assigned to a calculation.',
    'level': 'Level name.',
    'time': 'Time string.',
    'year': 'Year extracted from time string.',
    'month': 'Month extracted from time string.',
    'day': 'Day extracted from time string.',
    'cid': 'Unique identifier for a calculation name.',
    'value': 'Value associated with a variable or calculation.',
    'did': 'Dataset identifier see *_did.csv file for additional information on dataset requests.',
    'uri': 'Path to input data at execution time.',
    'alias': 'If not assigned, this will be the same as the variable name.'
}


class AbstractMetaConverter(AbstractConverter):
    """
    Base class for all metadata converters.

    :param ops: An OpenClimateGIS operations object.
    :type ops: :class:`ocgis.api.operations.OcgOperations`
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, ops):
        self.ops = ops


class MetaJSONConverter(AbstractMetaConverter):

    @classmethod
    def validate_ops(cls, ops):
        from ocgis.api.parms.definition import OutputFormat
        from ocgis import Field

        if len(ops.dataset) > 1:
            msg = 'Only one request dataset allowed for "{0}".'.format(constants.OUTPUT_FORMAT_METADATA_JSON)
            raise DefinitionValidationError(OutputFormat, msg)
        else:
            for element in ops.dataset.itervalues():
                if isinstance(element, Field):
                    msg = 'Fields may not be converted to "{0}".'.format(constants.OUTPUT_FORMAT_METADATA_JSON)
                    raise DefinitionValidationError(OutputFormat, msg)

    def write(self):
        driver = self.ops.dataset.first().driver
        """:type driver: :class:`ocgis.api.request.driver.base.AbstractDriver`"""
        return driver.get_source_metadata_as_json()


class MetaOCGISConverter(AbstractMetaConverter):
    _meta_filename = 'metadata.txt'

    def get_rows(self):
        lines = ['OpenClimateGIS v{0} Metadata File'.format(ocgis.__release__)]
        lines.append('  Generated (UTC): {0}'.format(datetime.datetime.utcnow()))
        lines.append('')
        if self.ops.output_format != 'meta':
            lines.append(
                'This is OpenClimateGIS-related metadata. Data-level metadata may be found in the file named: {0}'.format(
                    self.ops.prefix + '_source_metadata.txt'))
            lines.append('')
        lines.append('== Potential Header Names with Definitions ==')
        lines.append('')
        sh = sorted(HEADERS)
        for key in sh:
            msg = '  {0} :: {1}'.format(key.upper(), '\n'.join(justify_row(HEADERS[key]))).replace('::     ', ':: ')
            lines.append(msg)
        lines.append('')
        lines.append('== Argument Definitions and Content Descriptions ==')
        lines.append('')
        for v in sorted(self.ops.__dict__.itervalues()):
            if isinstance(v, AbstractParameter):
                lines.append(v.get_meta())

        # collapse lists
        ret = []
        for line in lines:
            if not isinstance(line, basestring):
                for item in line:
                    ret.append(item)
            else:
                ret.append(line)
        return ret

    def write(self):
        return '\n'.join(self.get_rows())
