import os
from collections import OrderedDict
from warnings import warn
import argparse

import fiona
from shapely.geometry.geo import shape, mapping

from ocgis.constants import OCGIS_UNIQUE_GEOMETRY_IDENTIFIER


class ShpProcess(object):
    """
    :param str path: Path to shapefile to process.
    :param out_folder: Path to the folder to write processed shapefiles to.
    """

    def __init__(self, path, out_folder):
        self.path = path
        self.out_folder = out_folder

    def process(self, key=None, ugid=None, name=None):
        """
        :param str key: The name of the new output shapefile.
        :param str ugid: The integer attribute to copy as the unique identifier.
        :param str name: The name of the unique identifer. If ``None``, defaults to
         :attr:`ocgis.constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER`.
        """

        # get the original shapefile file name
        original_name = os.path.split(self.path)[1]
        # get the new name if a key is passed
        if key is None:
            new_name = original_name
        else:
            new_name = key + '.shp'
        # the name of the new shapefile
        new_shp = os.path.join(self.out_folder, new_name)
        # update the schema to include UGID
        meta = self._get_meta_()

        identifier = name or OCGIS_UNIQUE_GEOMETRY_IDENTIFIER
        if identifier in meta['schema']['properties']:
            meta['schema']['properties'].pop(identifier)
        new_properties = OrderedDict({identifier: 'int'})
        new_properties.update(meta['schema']['properties'])
        meta['schema']['properties'] = new_properties
        ctr = 1
        with fiona.open(new_shp, 'w', **meta) as sink:
            for feature in self._iter_source_():
                if ugid is None:
                    feature['properties'].update({identifier: ctr})
                    ctr += 1
                else:
                    feature['properties'].update({identifier: int(feature['properties'][ugid])})
                sink.write(feature)

        # remove the cpg file. this raises many, many warnings on occasion
        # os.remove(new_shp.replace('.shp', '.cpg'))

        return new_shp

    def _get_meta_(self):
        with fiona.open(self.path, 'r') as source:
            return source.meta

    def _iter_source_(self):
        with fiona.open(self.path, 'r') as source:
            for feature in source:
                # ensure the feature is valid
                # https://github.com/Toblerity/Fiona/blob/master/examples/with-shapely.py
                geom = shape(feature['geometry'])
                try:
                    if not geom.is_valid:
                        clean = geom.buffer(0.0)
                        geom = clean
                        feature['geometry'] = mapping(geom)
                        assert clean.is_valid
                        assert (clean.geom_type == 'Polygon')
                except (AssertionError, AttributeError) as e:
                    warn('{2}. Invalid geometry found with id={0} and properties: {1}'.format(feature['id'],
                                                                                              feature['properties'],
                                                                                              e))
                feature['shapely'] = geom
                yield feature


def main(cargs):
    sp = ShpProcess(cargs.in_shp, cargs.folder)
    if cargs.folder is None:
        cargs.folder = os.getcwd()
    print(sp.process())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add ugid to shapefile')

    parser.add_argument('--ugid', help='name of ugid variable, default is None', default=None)
    parser.add_argument('--folder', help='path to the output folder', nargs='?')
    parser.add_argument('--key', help='optional new name for the shapefile', nargs=1, type=str)
    parser.add_argument('in_shp', help='path to input shapefile')

    parser.set_defaults(func=main)
    pargs = parser.parse_args()
    pargs.func(pargs)
