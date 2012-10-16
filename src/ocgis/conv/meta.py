import datetime
import ocgis


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
'value':'Value associated with a variable or calculation.'
}

class MetaConverter(object):
    
    def __init__(self,desc,uri=None):
        self.desc = desc
        self.uri = uri
        
    def write(self):
        from ocgis.api.interp.definition import DEF_ARGS
        
        lines = ['== OpenClimateGIS v{1} Metafile Generated (UTC): {0} =='.format(datetime.datetime.utcnow(),ocgis.__VER__)]
        lines.append('')
        if self.uri is not None:
            lines.append('Requested URL:')
            lines.append(self.uri)
            lines.append('')
        lines.append('++++ Parameter and Slug Definitions ++++')
        lines.append('')
        lines.append('Contains parameter descriptions for an OpenClimateGIS call based on operational dictionary values. The key/parameter names appears first in each "===" group with the name of URL-encoded slug name if it exists ("None" otherwise).')
        lines.append('')
        
        for Da in DEF_ARGS:
            obj = Da(self.desc.get(Da.name))
            msg = "{0}, URL slug name '{1}'".format(obj.name,obj.url_slug_name)
            divider = ''.join(['=' for ii in range(len(msg))])
            lines.append(divider)
            lines.append(msg)
            lines.append(divider)
            lines.append('')
            lines.append(obj.message())
            lines.append('')
        
        lines.append('++++ Potential Header Names and Definitions ++++')
        lines.append('')
        sh = sorted(HEADERS)
        for key in sh:
            msg = '{0} :: {1}'.format(key.upper(),HEADERS[key])
            lines.append(msg)
        ret = '\n'.join(lines)
        return(ret)