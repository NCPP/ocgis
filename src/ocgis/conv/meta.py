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
    
    def __init__(self,desc):
        self.desc = desc
        
    def write(self):
        from ocgis.api.interp.definition import DEF_ARGS
        
        print '== OpenClimateGIS v{1} Metafile Generated (UTC): {0} =='.format(datetime.datetime.utcnow(),ocgis.__VER__)
        print ''
        print '++++ Parameter and Slug Definitions ++++'
        print ''
        print 'Contains parameter descriptions for an OpenClimateGIS call based on operational dictionary values. The key/parameter names appears first in each "===" group with the name of URL-encoded slug name if it exists ("None" otherwise).'
        print ''
        for Da in DEF_ARGS:
            obj = Da(self.desc.get(Da.name))
            msg = "{0}, URL slug name '{1}'".format(obj.name,obj.url_slug_name)
            divider = ''.join(['=' for ii in range(len(msg))])
            print divider
            print msg
            print divider
            print ''
            print obj.message()
            print ''
        
        print '++++ Potential Header Names and Definitions ++++'
        print ''
        sh = sorted(HEADERS)
        for key in sh:
            msg = '{0} :: {1}'.format(key.upper(),HEADERS[key])
            print msg
        import ipdb;ipdb.set_trace()