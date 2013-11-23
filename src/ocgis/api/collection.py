from collections import OrderedDict
from ocgis.interface.base.crs import CFWGS84
from ocgis import constants
from ocgis.util.logging_ocgis import ocgis_lh


class SpatialCollection(OrderedDict):
    _default_headers = constants.raw_headers
    
    def __init__(self,meta=None,key=None,crs=None,headers=None,value_keys=None):
        self.meta = meta
        self.key = key
        self.crs = crs or CFWGS84()
        self.headers = headers or self._default_headers
        self.value_keys = value_keys
    
        self.geoms = {}
        self.properties = OrderedDict()
        
        self._uid_ctr_field = 1

        super(SpatialCollection,self).__init__()
        
    @property
    def _archetype_field(self):
        ukey = self.keys()[0]
        fkey = self[ukey].keys()[0]
        return(self[ukey][fkey])
        
    def add_field(self,ugid,geom,alias,field,properties=None):
        ## add field unique identifier if it does not exist
        try:
            if field.uid is None:
                field.uid = self._uid_ctr_field
                self._uid_ctr_field += 1
        ## likely a nonetype from an empty subset
        except AttributeError as e:
            if field is None:
                pass
            else:
                ocgis_lh(exc=e,loggger='collection')
            
        self.geoms.update({ugid:geom})
        self.properties.update({ugid:properties})
        if ugid not in self:
            self.update({ugid:{}})
        assert(alias not in self[ugid])
        self[ugid].update({alias:field})
                
    def get_iter_dict(self,use_upper_keys=False,conversion_map=None):
        r_headers = self.headers
        use_conversion = False if conversion_map is None else True
        for ugid,field_dict in self.iteritems():
            for field in field_dict.itervalues():
                for row in field.get_iter(value_keys=self.value_keys):
                    row['ugid'] = ugid
                    yld_row = {k:row[k] for k in r_headers}
                    if use_conversion:
                        for k,v in conversion_map.iteritems():
                            yld_row[k] = v(yld_row[k])
                    if use_upper_keys:
                        yld_row = {k.upper():v for k,v in yld_row.iteritems()}
                    yield(row['geom'],yld_row)
                    
    def get_iter_elements(self):
        for ugid,fields in self.iteritems():
            for field_alias,field in fields.iteritems():
                for var_alias,variable in field.variables.iteritems():
                    yield(ugid,field_alias,var_alias,variable)
                
    def gvu(self,ugid,alias_variable,alias_field=None):
        ref = self[ugid]
        if alias_field is None:
            field = ref.values()[0]
        else:
            field = ref[alias_field]
        return(field.variables[alias_variable].value)
