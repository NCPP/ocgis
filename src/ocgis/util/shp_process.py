import ocgis
import fiona
from shapely.geometry.geo import shape, mapping
import os
from collections import OrderedDict
import shutil
from warnings import warn


class ShpProcess(object):
    
    def __init__(self,path):
        self.path = path
        
    def process(self,out_folder,key,ugid=None):
        ## make the output folder
        new_folder = os.path.join(out_folder,key)
        os.mkdir(new_folder)
        try:
            ## the name of the new shapefile
            new_shp = os.path.join(new_folder,key+'.shp')
            ## update the schema to include UGID
            meta = self._get_meta_()
            if 'UGID' in meta['schema']['properties']:
                meta['schema']['properties'].pop('UGID')
            new_properties = OrderedDict({'UGID':'int'})
            new_properties.update(meta['schema']['properties'])
            meta['schema']['properties'] = new_properties
            ctr = 1
            with fiona.open(new_shp, 'w',**meta) as sink:
                for feature in self._iter_source_():
                    if ugid is None:
                        feature['properties'].update({'UGID':ctr})
                        ctr += 1
                    else:
                        feature['properties'].update({'UGID':int(feature['properties'][ugid])})
                    sink.write(feature)
            ## remove the cpg file. this raises many, many warnings on occasion
            os.remove(new_shp.replace('.shp','.cpg'))
        except:
            ## remove the created folder on an exception
            shutil.rmtree(new_folder)
            raise
                
    def _get_meta_(self):
        with fiona.open(self.path,'r') as source:
            return(source.meta)
        
    def _iter_source_(self):
        with fiona.open(self.path,'r') as source:
            for feature in source:
                ## ensure the feature is valid
                ## https://github.com/Toblerity/Fiona/blob/master/examples/with-shapely.py
                try:
                    geom = shape(feature['geometry'])
                    if not geom.is_valid:
                        clean = geom.buffer(0.0)
                        geom = clean
                        feature['geometry'] = mapping(geom)
                        assert(clean.is_valid)
                        assert(clean.geom_type == 'Polygon')
                except (AssertionError,AttributeError) as e:
                    warn('{2}. Invalid geometry found with id={0} and properties: {1}'.format(feature['id'],
                                                                                                feature['properties'],
                                                                                                                e))
                feature['shapely'] = geom
                yield(feature)
                
                


#################################################################################
#
#config = ConfigParser.ConfigParser()
#config.read('setup.cfg')
#
#parser = argparse.ArgumentParser(description='install/uninstall OpenClimateGIS. use "setup.cfg" to find or set default values.')
#parser.add_argument('-v','--verbose',action='store_true',help='print potentially useful information')
#subparsers = parser.add_subparsers()
#
#pinstall = subparsers.add_parser('install',help='install the OpenClimateGIS Python package')
#pinstall.set_defaults(func=install)
#
#pubuntu = subparsers.add_parser('install_dependencies_ubuntu',help='attempt to install OpenClimateGIS dependencies using standard Ubuntu Linux operations')
#pubuntu.set_defaults(func=install_dependencies_ubuntu)
#
#puninstall = subparsers.add_parser('uninstall',help='instructions on how to uninstall the OpenClimateGIS Python package')
#puninstall.set_defaults(func=uninstall)
#
#ppackage = subparsers.add_parser('package',help='utilities for packaging shapefile and NetCDF test datasets')
#ppackage.set_defaults(func=package)
#ppackage.add_argument('target',type=str,choices=['shp','nc','all'],help='Select the files to package.')
#ppackage.add_argument('-d','--directory',dest='d',type=str,metavar='dir',help='the destination directory. if not specified, it defaults to the current working directory.')
#
#pargs = parser.parse_args()
#pargs.func(pargs)