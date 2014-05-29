import abc
from collections import OrderedDict
from ocgis.interface.base.crs import CFWGS84
from ocgis import constants
from ocgis.util.logging_ocgis import ocgis_lh


class AbstractCollection(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._storage = OrderedDict()
        self._storage_id = []

    @property
    def _storage_id_next(self):
        try:
            ret = max(self._storage_id) + 1
        ## max of an empty list
        except ValueError:
            if len(self._storage_id) == 0:
                ret = 1
            else:
                raise
        return ret

    def __contains__(self, item):
        return item in self._storage

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            ret = self.__dict__ == other.__dict__
        else:
            ret = False
        return ret

    def __iter__(self):
        for key in self.iterkeys():
            yield key

    def __getitem__(self, item):
        return self._storage[item]

    def __len__(self):
        return len(self._storage)

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __repr__(self):
        ret = '{0}({1})'.format(self.__class__.__name__, [(k, v) for k, v in self.iteritems()])
        return ret

    def __str__(self):
        return self.__repr__()

    def first(self):
        for key in self.iterkeys():
            return self._storage[key]

    def items(self):
        return self._storage.items()

    def iteritems(self):
        for k, v in self._storage.iteritems():
            yield k, v

    def iterkeys(self):
        for k in self._storage.iterkeys():
            yield k

    def itervalues(self):
        for v in self._storage.itervalues():
            yield v

    def keys(self):
        return self._storage.keys()

    def update(self, dictionary):
        self._storage.update(dictionary)

    def values(self):
        return self._storage.values()


class SpatialCollection(AbstractCollection):
    _default_headers = constants.raw_headers
    
    def __init__(self, meta=None, key=None, crs=None, headers=None, value_keys=None):
        super(SpatialCollection, self).__init__()

        self.meta = meta
        self.key = key
        self.crs = crs or CFWGS84()
        self.headers = headers or self._default_headers
        self.value_keys = value_keys
    
        self.geoms = OrderedDict()
        self.properties = OrderedDict()
        
        # self._uid_ctr_field = 1
        # self._ugid = OrderedDict()
        
    @property
    def _archetype_field(self):
        ukey = self.keys()[0]
        fkey = self[ukey].keys()[0]
        return(self[ukey][fkey])
        
    def add_field(self, ugid, geom, field, properties=None, name=None):
        """
        :param int ugid:
        :param :class:`shapely.Geometry`:
        :param :class:`ocgis.Field`:
        :param dict properties:
        :param str name:
        """
        name = name or field.name

        ## add field unique identifier if it does not exist
        try:
            if field.uid is None:
                field.uid = self._storage_id_next
                self._storage_id.append(field.uid)
        ## likely a nonetype from an empty subset
        except AttributeError as e:
            if field is None:
                pass
            else:
                ocgis_lh(exc=e, logger='collection')
            
        self.geoms.update({ugid:geom})
        self.properties.update({ugid:properties})
        if ugid not in self:
            self.update({ugid:{}})
        assert(name not in self[ugid])
        self[ugid].update({name:field})
                
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
                    
    def get_iter_melted(self):
        for ugid,container in self.iteritems():
            for field_alias,field in container.iteritems():
                for variable_alias,variable in field.variables.iteritems():
                    yield(dict(ugid=ugid,field_alias=field_alias,field=field,variable_alias=variable_alias,variable=variable))
                
    def gvu(self,ugid,alias_variable,alias_field=None):
        ref = self[ugid]
        if alias_field is None:
            field = ref.values()[0]
        else:
            field = ref[alias_field]
        return(field.variables[alias_variable].value)
