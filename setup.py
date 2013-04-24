from distutils.core import setup
import sys
import os
import argparse
import ConfigParser
from subprocess import check_call
import shutil
import os
import tempfile
import tarfile


def install(pargs,version='0.05b-dev'):
    ## check python version
    python_version = float(sys.version_info[0]) + float(sys.version_info[1])/10
    if python_version != 2.7:
        raise(ImportError(
            'This software requires Python version 2.7.x. You have {0}.x'.format(python_version)))

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
          url='http://ncpp.github.io/ocgis/install.html#installing-openclimategis',
          license='NCSA License',
          platforms=['all'],
          packages=packages,
          package_dir=package_dir
          )

def install_dependencies_ubuntu(pargs):

    cwd = os.getcwd()
    out = 'install_out.log'
    err = 'install_err.log'
    odir = tempfile.mkdtemp()
    stdout = open(out,'w')
    stderr = open(err,'w')

    def call(args):
        check_call(args,stdout=stdout,stderr=stderr)

    def install_dependency(odir,url,tarball,edir,config_flags=None,custom_make=None):
        path = tempfile.mkdtemp(dir=odir)
        os.chdir(path)
        print('downloading {0}...'.format(edir))
        call(['wget',url])
        print('extracting {0}...'.format(edir))
        call(['tar','-xzvf',tarball])
        os.chdir(edir)
        if custom_make is None:
            print('configuring {0}...'.format(edir))
            call(['./configure']+config_flags)
            print('building {0}...'.format(edir))
            call(['make'])
            print('installing {0}...'.format(edir))
            call(['make','install'])
        else:
            print('installing {0}...'.format(edir))
            custom_make()

    print('installing apt packages...')
    call(['apt-get','update'])
    call(['apt-get','-y','install','g++','libz-dev','curl','wget','python-dev','python-setuptools','python-gdal'])
    print('installing shapely...')
    call(['easy_install','shapely'])

    prefix = '/usr/local'

    hdf5 = 'hdf5-1.8.10-patch1'
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

    nc4p = 'netCDF4-1.0.4'
    nc4p_tarball = '{0}.tar.gz'.format(nc4p)
    nc4p_url = 'http://netcdf4-python.googlecode.com/files/{0}'.format(nc4p_tarball)
    call(['ldconfig'])
    def nc4p_make():
        call(['python','setup.py','install'])
    install_dependency(odir,nc4p_url,nc4p_tarball,nc4p,custom_make=nc4p_make)


    stdout.close()
    stderr.close()
    #shutil.rmtree(odir)
    os.chdir(cwd)
    print('dependencies installed.')
    
def package(pargs):
    ## get destination directory
    dst = pargs.d or os.getcwd()
    if not os.path.exists(dst):
        raise(IOError('Destination directory does not exist: {0}'.format(dst)))

    ## import ocgis using relative locations
    opath = os.path.join(os.getcwd(),'src')
    sys.path.append(opath)
    
    to_tar = []
    
    if pargs.target in ['shp','all']:
        import ocgis
        shp_dir = ocgis.env.DIR_SHPCABINET
        for dirpath,dirnames,filenames in os.walk(shp_dir):
            for filename in filenames:
                path = os.path.join(dirpath,filename)
                arcname = os.path.join('ocgis_data','shp',os.path.split(dirpath)[1],filename)
                to_tar.append({'path':path,'arcname':arcname})
    
    if pargs.target in ['nc','all']:
        from ocgis.test.base import TestBase
        tdata = TestBase.get_tdata()
        for key,value in tdata.iteritems():
            path = tdata.get_uri(key)
            arcname = os.path.join('ocgis_data','nc',tdata.get_relative_path(key))
            to_tar.append({'path':path,'arcname':arcname})
    
    out = os.path.join(os.path.join(dst,'ocgis_data.tar.gz'))
    if pargs.verbose: print('destination file is: {0}'.format(out))
    tf = tarfile.open(out,'w:gz')
    try:
        for tz in to_tar:
            if pargs.verbose and any([tz['path'].endswith(ii) for ii in ['shp','nc']]):
                print('adding: {0}'.format(tz['path']))
            tf.add(tz['path'],arcname=tz['arcname'])
    finally:
        if pargs.verbose: print('closing file...')
        tf.close()
    
    if pargs.verbose: print('compression complete.')

def uninstall(pargs):
    try:
        import ocgis
        print('To uninstall, manually remove the Python package folder located here: {0}'.format(os.path.split(ocgis.__file__)[0]))
    except ImportError:
        raise(ImportError("Either OpenClimateGIS is not installed or not available on the Python path."))

################################################################################

config = ConfigParser.ConfigParser()
config.read('setup.cfg')

parser = argparse.ArgumentParser(description='install/uninstall OpenClimateGIS. use "setup.cfg" to find or set default values.')
parser.add_argument('-v','--verbose',action='store_true',help='print potentially useful information')
subparsers = parser.add_subparsers()

pinstall = subparsers.add_parser('install',help='install the OpenClimateGIS Python package')
pinstall.set_defaults(func=install)

pubuntu = subparsers.add_parser('install_dependencies_ubuntu',help='attempt to install OpenClimateGIS dependencies using standard Ubuntu Linux operations')
pubuntu.set_defaults(func=install_dependencies_ubuntu)

puninstall = subparsers.add_parser('uninstall',help='instructions on how to uninstall the OpenClimateGIS Python package')
puninstall.set_defaults(func=uninstall)

ppackage = subparsers.add_parser('package',help='utilities for packaging shapefile and NetCDF test datasets')
ppackage.set_defaults(func=package)
ppackage.add_argument('target',type=str,choices=['shp','nc','all'],help='Select the files to package.')
ppackage.add_argument('-d','--directory',dest='d',type=str,metavar='dir',help='the destination directory. if not specified, it defaults to the current working directory.')

pargs = parser.parse_args()
pargs.func(pargs)
