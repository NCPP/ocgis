from distutils.core import setup
import sys
import os
import argparse
import ConfigParser

        
config = ConfigParser.ConfigParser()
config.read('setup.cfg')

parser = argparse.ArgumentParser(description='Install/uninstall OpenClimateGIS. Use "setup.cfg" to find or set default values.')
parser.add_argument("action",type=str,choices=['install','uninstall'],help='action to perform with the installer')
#parser.add_argument("--with-shp",help='download shapefile regions of interest',action='store_true')
#parser.add_argument("--shp-prefix",help='location to hold shapefiles',default=config.get('shp','url'))
#parser.add_argument("--shp-url",help='URL location of shapefiles',default=config.get('shp','url'))
#parser.add_argument("--with-bin",help='download binary files for testing',default=config.get('test','dir'),action='store_true')
#parser.add_argument("--bin-prefix",help='location to hold binary test files',default=config.get('shp','url'))
#parser.add_argument("--bin-url",help='URL location of binary test files',default=config.get('test','url'))
args = parser.parse_args()

################################################################################

def install(version='0.04.01b'):
    ## check python version
    python_version = float(sys.version_info.major) + float(sys.version_info.minor)/10
    if python_version != 2.7:
        raise(ImportError('This software requires Python version 2.7.'))
    
    ## attempt package imports
    pkgs = ['numpy','netCDF4','osgeo','shapely']
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            msg = 'Unable to import Python package: "{0}".'.format(pkg)
            raise(ImportError(msg))
    
    ## get package structure
    def _get_dot_(path,root='src'):
        ret = []
        path_parse = path
        while True:
            path_parse,tail = os.path.split(path_parse)
            if tail == root:
                break
            else:
                ret.append(tail)
        ret.reverse()
        return('.'.join(ret))
    package_dir = {'':'src'}
    src_path = os.path.join(package_dir.keys()[0],package_dir.values()[0],'ocgis')
    packages = []
    for dirpath,dirnames,filenames in os.walk(src_path):
        if '__init__.py' in filenames:
            package = _get_dot_(dirpath)
            packages.append(package)
    
    ## run the installation
    setup(name='ocgis',
          version=version,
          author='NESII/CIRES/NOAA-ESRL',
          author_email='ocgis_support@list.woc.noaa.gov',
          url='https://github.com/NCPP/ocgis/tags',
          license='BSD License',
          platforms=['all'],
          packages=packages,
          package_dir=package_dir
          )

def uninstall():
    try:
        import ocgis
        print('To uninstall, manually remove the Python package folder located here: {0}'.format(os.path.split(ocgis.__file__)[0]))
    except ImportError:
        raise(ImportError("Either OpenClimateGIS is not installed or not available on the Python path."))

################################################################################

if args.action == 'install':
    install()
elif args.action == 'uninstall':
    uninstall()