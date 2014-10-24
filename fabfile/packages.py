import os

from fabric.context_managers import cd, shell_env, quiet, path
from fabric.contrib.files import append
from fabric.operations import sudo, run, require, prompt

from base import AbstractMakeInstaller, AbstractPipInstaller, AbstractSetupInstaller, AbstractInstaller

from helpers import fcmd, parser


class NumpyInstaller(AbstractPipInstaller):
    prefix = 'numpy'
    apt_packages = ['python-pip', 'python-dev']


class HDF5Installer(AbstractMakeInstaller):
    prefix = 'hdf5'
    apt_packages = ['zlib1g-dev']
    configure_options = ['--enable-shared', '--enable-hl']
    template_wget_url = 'http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-{version}/src/hdf5-{version}.tar.gz'
    template_uncompressed_dir = 'hdf5-{version}'


class NetCDF4Installer(AbstractMakeInstaller):
    prefix = 'netcdf4'
    apt_packages = ['libcurl4-openssl-dev']
    configure_options = ['--enable-netcdf-4', '--enable-dap', '--enable-utilities']
    template_wget_url = 'ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-{version}.tar.gz'
    template_uncompressed_dir = 'netcdf-{version}'

    def __init__(self, version, hdf5):
        super(NetCDF4Installer, self).__init__(version)
        self.hdf5 = hdf5

    def configure(self):
        kwds = {'CPPFLAGS': '-I{0}'.format(os.path.join(self.hdf5.path_install, 'include')),
                'LDFLAGS': '-L{0}'.format(os.path.join(self.hdf5.path_install, 'lib'))}
        with shell_env(**kwds):
            super(NetCDF4Installer, self).configure()


class NetCDF4PythonInstaller(AbstractPipInstaller):
    prefix = 'netCDF4'

    def __init__(self, version, netcdf4):
        super(NetCDF4PythonInstaller, self).__init__(version)
        self.netcdf4 = netcdf4

    def install(self):
        kwds = {'HDF5_DIR': self.netcdf4.hdf5.path_install,
                'NETCDF4_DIR': self.netcdf4.path_install}
        with shell_env(**kwds):
            super(NetCDF4PythonInstaller, self).install()


class GeosInstaller(AbstractMakeInstaller):
    prefix = 'geos'
    template_wget_url = 'http://download.osgeo.org/geos/geos-{version}.tar.bz2'
    template_uncompressed_dir = 'geos-{version}'

    @staticmethod
    def uncompress(fn):
        cmd = ['tar', '-xjf', fn]
        fcmd(run, cmd)


class Proj4Installer(AbstractMakeInstaller):
    prefix = 'proj4'
    template_wget_url = 'http://download.osgeo.org/proj/proj-{version}.tar.gz'
    template_wget_url_datum_grid = 'http://download.osgeo.org/proj/proj-datumgrid-{0}.tar.gz'
    template_uncompressed_dir = 'proj-{version}'

    def __init__(self, version, version_datum_grid):
        super(Proj4Installer, self).__init__(version)
        self.version_datum_grid = version_datum_grid

    def initialize(self):
        super(Proj4Installer, self).initialize()
        with cd(self.path_src):
            wget_url_datum_grid = self.template_wget_url_datum_grid.format(self.version_datum_grid)
            cmd = ['wget', wget_url_datum_grid]
            fcmd(run, cmd)
            cmd = ['tar', '-xzvf', os.path.split(wget_url_datum_grid)[1], '-C', os.path.join(self.path_uncompressed_dir, 'nad')]
            fcmd(run, cmd)


class GDALInstaller(AbstractMakeInstaller):
    prefix = 'gdal'
    configure_options = ['--with-python']
    template_wget_url = 'http://download.osgeo.org/gdal/{version}/gdal-{version}.tar.gz'
    template_uncompressed_dir = 'gdal-{version}'
    make_check = False

    def __init__(self, version, geos, proj4):
        super(GDALInstaller, self).__init__(version)
        self.geos = geos
        self.proj4 = proj4
        self._ld_preload = False

    @property
    def configure_cmd(self):
        ret = super(GDALInstaller, self).configure_cmd
        with_geos = os.path.join(self.geos.path_install, 'bin', 'geos-config')
        static_proj4 = self.proj4.path_install
        ret += ['--with-geos={0}'.format(with_geos), '--with-static-proj4={0}'.format(static_proj4)]
        return ret

    def finalize(self):
        try:
            with quiet():
                require('LD_PRELOAD')
            raise NotImplementedError
        except SystemExit:
            # add the gdal library to the LD_PRELOAD_PATH to ensure osgeo may find the shared library
            lib_path = os.path.join(self.path_install, 'lib', 'libgdal.so.1')
            append('~/.bashrc', '\n# for some reason, the python osgeo library needs this to link properly')
            append('~/.bashrc', 'export LD_PRELOAD={0}\n'.format(lib_path))
            prompt('The .bashrc file on the remote server needs to be sourced before osgeo is available. Press Enter to continue.')


class CythonInstaller(AbstractPipInstaller):
    prefix = 'cython'


class ShapelyInstaller(AbstractPipInstaller):
    prefix = 'shapely'

    def __init__(self, version, geos):
        self.version = version
        self.geos = geos

    def install(self):
        kwds = {'CPPFLAGS': '-I{0}'.format(os.path.join(self.geos.path_install, 'include')),
                'LDFLAGS': '-L{0}'.format(os.path.join(self.geos.path_install, 'lib'))}
        with shell_env(**kwds):
            super(ShapelyInstaller, self).install()


class FionaInstaller(AbstractPipInstaller):
    prefix = 'fiona'

    def __init__(self, version, gdal):
        self.version = version
        self.gdal = gdal

    def install(self):
        kwds = {'CPPFLAGS': '-I{0}'.format(os.path.join(self.gdal.path_install, 'include')),
                # 'LDFLAGS': '-L{0}'.format(os.path.join(self.geos.path_install, 'lib'))
        }
        with path(os.path.join(self.gdal.path_install, 'bin')):
            with shell_env(**kwds):
                super(FionaInstaller, self).install()

    def validate(self):
        """Requires the LD_PRELOAD path to gdal/lib/libgdal.so.1 be set."""


class RtreeInstaller(AbstractPipInstaller):
    prefix = 'rtree'
    apt_packages = ['libspatialindex-dev']


class CFUnitsInstaller(AbstractSetupInstaller):
    prefix = 'cfunits'
    apt_packages = ['libudunits2-0']
    template_wget_url = 'https://cfunits-python.googlecode.com/files/cfunits-{version}.tar.gz'
    template_uncompressed_dir = 'cfunits-{version}'
    cfunits_install_dir = '/usr/local/lib/python2.7/dist-packages/cfunits'

    def finalize(self):
        src = os.path.join(self.path_configure, 'cfunits', 'etc')
        cmd = ['cp', '-r', src, self.cfunits_install_dir]
        fcmd(sudo, cmd)


class ESMFInstaller(AbstractMakeInstaller):
    prefix = 'esmf'
    apt_packages = ['gfortran', 'g++']
    template_esmf_targz = '/home/ubuntu/htmp/esmf_{version}_src.tar.gz'
    template_uncompressed_dir = 'esmf'
    template_wget_url = None

    @property
    def path_esmf_targz(self):
        return self.template_esmf_targz.format(version=self.version.replace('.', '_'))

    def initialize(self):
        AbstractInstaller.initialize(self)
        self._make_dir_src_()
        with cd(self.path_src):
            cmd = ['cp', self.path_esmf_targz, self.path_src]
            fcmd(run, cmd)
            tar_name = os.path.split(self.path_esmf_targz)[1]
            self.uncompress(tar_name)

    def install(self):
        kwds = {'ESMF_DIR': self.path_configure,
                'ESMF_INSTALL_PREFIX': self.path_install,
                'ESMF_INSTALL_LIBDIR': os.path.join(self.path_install, 'lib')}
        with shell_env(**kwds):
            with cd(self.path_configure):
                cmd = ['make', '-j', self.j]
                fcmd(run, cmd)
                cmd = ['sudo', '-E', 'make', 'install']
                fcmd(run, cmd)


class ESMPyInstaller(AbstractInstaller):
    prefix = 'ESMF'

    def __init__(self, esmf):
        super(ESMPyInstaller, self).__init__(esmf.version)
        self.esmf = esmf

    def install(self):
        path_esmpy_src = os.path.join(self.esmf.path_src, 'esmf', 'src', 'addon', 'ESMPy')
        esmfmkfile = os.path.join(self.esmf.path_install, 'lib', 'esmf.mk')
        with cd(path_esmpy_src):
            cmd = ['python', 'setup.py', 'build', '--ESMFMKFILE={0}'.format(esmfmkfile)]
            fcmd(run, cmd)
            cmd = ['python', 'setup.py', 'install']
            fcmd(sudo, cmd)

    def validate(self):
        cmd = ['python', '-c', '"import {0}"'.format(self.prefix)]
        fcmd(run, cmd)


class IcclimInstaller(AbstractInstaller):
    prefix = 'icclim'
    git_url = 'https://github.com/tatarinova/icclim.git'
    apt_packages = ['git']

    def __init__(self, branch_name='master'):
        self.branch_name = branch_name

    def install(self):
        with cd(parser.get('server', 'dir_clone')):
            cmd = ['git', 'clone', self.git_url]
            fcmd(run, cmd)
            with cd(self.prefix):
                run('git checkout {0}'.format(self.branch_name))
                run('git pull')
                sudo('gcc -fPIC -g -c -Wall ./icclim/libC.c -o ./icclim/libC.o')
                sudo('gcc -shared -o ./icclim/libC.so ./icclim/libC.o')
                sudo('python setup.py install')

    def validate(self):
        cmd = ['python', '-c', '"import {0}"'.format(self.prefix)]
        fcmd(run, cmd)
