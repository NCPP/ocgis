from distutils.core import setup
import sys
import os
import argparse
import ConfigParser
from subprocess import check_call
import shutil
import os
import tempfile


config = ConfigParser.ConfigParser()
config.read('setup.cfg')

parser = argparse.ArgumentParser(description='Install/uninstall OpenClimateGIS. Use "setup.cfg" to find or set default values.')
parser.add_argument("action",type=str,choices=['install','install_all','uninstall'],help='The action to perform with the installer.')
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
    
def install_all():
    
    cwd = os.getcwd()
    out = 'install.log'
    odir = tempfile.mkdtemp()
    print(odir)
    stdout = open(out,'w')
    
    def call(args):
        check_call(args,stdout=stdout)
    
    def install_dependency(odir,url,tarball,edir,config_flags=None,custom_make=None):
        path = tempfile.mkdtemp(dir=odir)
        os.mkdir(path)
        os.chdir(path)
        call(['wget',url])
        call(['tar','-xzvf',tarball])
        os.chdir(edir)
        if custom_make is None:
            call(['./configure']+config_flags)
            call(['make'])
            call(['make install'])
        else:
            custom_make()
    
    call(['apt-get','update'])
    call(['apt-get','-y','install','g++','libz-dev','curl','wget','python-dev','python-setuptools','python-gdal'])
    call(['easy_install','shapely'])
    
    prefix = '/usr/local'
    
    hdf5 = 'hdf5-1.8.1.10-patch1'
    hdf5_tarball = '{0}.tar.gz'.format(hdf5)
    hdf5_url = 'http://www.hdfgroup.org/ftp/HDF5/current/src/{0}'.format(hdf5_tarball)
    hdf5_flags = ['--prefix={0}'.format(prefix),'--enable-shared','--enable-hl']
    install_dependency(odir,hdf5_url,hdf5_tarball,hdf5,hdf5_flags)
    
    nc4 = 'netcdf-4.2.1'
    nc4_tarball = '{0}.tar.gz'.format(nc4)
    nc4_url = 'ftp://ftp.unidata.ucar.edu/pub/netcdf/{0}'.format(nc4_tarball)
    nc4_flags = ['--prefix={0}'.format(prefix),'--enable-shared','--enable-dap','--enable-netcdf-4']
    os.putenv('LDFLAGS','-L{0}/lib'.format(prefix))
    os.putenv('CPPFLAGS','-I{0}/include'.format(prefix))
    install_dependency(odir,nc4_url,nc4_tarball,nc4,nc4_flags)
    os.unsetenv('LDFLAGS')
    os.unsetenv('CPPFLAGS')
    
    nc4p = 'netCDF4-1.0.2'
    nc4p_tarball = '{0}.tar.gz'.format(nc4p)
    nc4p_url = 'http://netcdf4-python.googlecode.com/files/{0}'.format(nc4p_tarball)
    call(['ldconfig'])
    def nc4p_make():
        call(['python','setup.py','install'])
    install_dependency(odir,nc4p_url,nc4p_tarball,nc4p,custom_make=nc4p_make)
    
    
    stdout.close()
    #shutil.rmtree(odir)
    os.chdir(cwd)
    
    install()

def uninstall():
    try:
        import ocgis
        print('To uninstall, manually remove the Python package folder located here: {0}'.format(os.path.split(ocgis.__file__)[0]))
    except ImportError:
        raise(ImportError("Either OpenClimateGIS is not installed or not available on the Python path."))

################################################################################

if args.action == 'install':
    install()
elif args.action == 'install_all':
    install_all()
elif args.action == 'uninstall':
    uninstall()