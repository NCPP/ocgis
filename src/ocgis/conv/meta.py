import datetime
import ocgis
from ocgis.api.parms.base import OcgParameter
from ocgis.util.justify import justify_row


HEADERS = {
'ugid':'User geometry identifier pulled from a provided set of selection geometries. Reduces to "1" for the case of no provided geometry.',
'gid':'Geometry identifier assigned by OpenClimateGIS to a dataset geometry. In the case of "aggregate=True" this is equivalent to "UGID".',
'tid':'Unique time identifier.',
'tgid':'Unique grouped time identifier.',
'vid':'Unique variable identifier.',
'lid':'Level identifier unique within a variable.',
'vlid':'Globally unique level identifier.',
'variable':'Name of request variable.',
'calc_name':'User-supplied name for a calculation.',
'level':'Level name.',
'time':'Time string.',
'year':'Year extracted from time string.',
'month':'Month extracted from time string.',
'day':'Day extracted from time string.',
'cid':'Unique identifier for a calculation name.',
'value':'Value associated with a variable or calculation.',
'did':'Dataset identifier see *_did.csv file for additional information on dataset requests.',
'uri':'Path to input data at execution time.',
'alias':'Dataset alias - may be equivalent to VARIABLE.'
}


class MetaConverter(object):
    _meta_filename = 'metadata.txt'
    
    def __init__(self,ops):
        self.ops = ops
        
    def get_rows(self):
        lines = ['OpenClimateGIS v{0} Metadata File'.format(ocgis.__release__)]
        lines.append('  Generated (UTC): {0}'.format(datetime.datetime.utcnow()))
        lines.append('')
        if self.ops.output_format != 'meta':
            lines.append('This is OpenClimateGIS-related metadata. Data-level metadata may be found in the file named: {0}'.format(self.ops.prefix+'_source_metadata.txt'))
            lines.append('')
        lines.append('== Potential Header Names with Definitions ==')
        lines.append('')
        sh = sorted(HEADERS)
        for key in sh:
            msg = '  {0} :: {1}'.format(key.upper(),'\n'.join(justify_row(HEADERS[key]))).replace('::     ',':: ')
            lines.append(msg)
        lines.append('')
        lines.append('== Argument Definitions and Content Descriptions ==')
        lines.append('')
        for v in sorted(self.ops.__dict__.itervalues()):
            if isinstance(v,OcgParameter):
                lines.append(v.get_meta())
            
        ## collapse lists
        ret = []
        for line in lines:
            if not isinstance(line,basestring):
                for item in line:
                    ret.append(item)
            else:
                ret.append(line)
        return(ret)

    @classmethod
    def validate_ops(cls, ops):
        """
        Validate an operations object.

        :param ops: The input operations object to validate.
        :type ops: :class:`ocgis.OcgOperations`
        :raises: DefinitionValidationError
        """
        
    def write(self):
        return('\n'.join(self.get_rows()))
