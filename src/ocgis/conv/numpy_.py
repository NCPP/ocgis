from ocgis.conv.base import AbstractConverter
from ocgis.api.collection import SpatialCollection


class NumpyConverter(AbstractConverter):
    _create_directory = False

    def __iter__(self):
        for coll in self.colls:
            yield coll

    def write(self):
        build = True
        for coll in self:
            if build:
                ret = SpatialCollection(meta=coll.meta, key=coll.key, crs=coll.crs, headers=coll.headers)
                build = False
            for k, v in coll.iteritems():
                field = v.values()[0]
                if field is None:
                    name = v.keys()[0]
                else:
                    name = None
                ret.add_field(v.values()[0], ugeom=coll.ugeom[k], name=name)

        return ret

    def _write_(self):
        pass