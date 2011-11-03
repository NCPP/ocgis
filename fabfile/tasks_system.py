from fabric.api import task
from fabric.api import run, sudo
from fabric.api import cd, settings, prefix

# build parameters
MAKEJOBS = '--jobs=4 ' # specify how many make commands to run simultaneously

SRCDIR = '~/src'

PROJ_VER = '4.7.0'
PROJ_SRC = SRCDIR +'/proj/' + PROJ_VER
PROJ_DIR = '/usr/local/proj/' + PROJ_VER

GEOS_VER = '3.2.2'
GEOS_SRC = SRCDIR + '/geos/' + GEOS_VER
GEOS_DIR = '/usr/local/geos/' + GEOS_VER

GDAL_VER = '1.8.1'
GDAL_SRC = SRCDIR + '/gdal/' + GDAL_VER
GDAL_DIR = '/usr/local/gdal/' + GDAL_VER

HDF5_VER = '1.8.7'
HDF5_SRC = SRCDIR + '/hdf5/' + HDF5_VER
HDF5_DIR = '/usr/local/hdf5/' + HDF5_VER

NETCDF4_VER = '4.1.1'
NETCDF4_SRC = SRCDIR + '/hdf5/' + NETCDF4_VER
NETCDF4_DIR = '/usr/local/netCDF4/' + NETCDF4_VER


@task(default=True)
def install_system_dependencies():
    '''Installs required geospatial libraries'''
    update_system()
    install_build_dependencies()
    create_source_code_folder()
    install_proj()
    install_geos()
    install_gdal()
    install_hdf5()
    install_netcdf4()
    install_python_dependencies()

@task
def update_system():
    '''Update the list of Ubuntu packages'''
    sudo('apt-get -y update')
    sudo('apt-get -y upgrade')

@task
def install_build_dependencies():
    '''Install dependencies for building libraries'''
    sudo('apt-get -y install wget')
    sudo('apt-get -y install unzip')
    sudo('apt-get -y install gcc')
    sudo('apt-get -y install g++')
    sudo('apt-get -y install python-dev')
    sudo('apt-get -y install python-setuptools')
    sudo('apt-get -y install python-pip')
    sudo('apt-get -y install swig')
    sudo('apt-get -y install git-core')
    sudo('apt-get -y install mercurial')

@task
def create_source_code_folder():
    '''Creates a source code folder'''
    with settings(warn_only=True):
        run('mkdir {0}'.format(SRCDIR))

@task
def install_proj():
    '''Install Proj.4'''
    run('mkdir -p {projsrc}'.format(projsrc=PROJ_SRC))
    with cd(PROJ_SRC):
        run('wget http://download.osgeo.org/proj/proj-datumgrid-1.5.zip')
        run('wget http://download.osgeo.org/proj/proj-{projver}.tar.gz'.format(projver=PROJ_VER))
        run('tar xzf proj-{projver}.tar.gz'.format(projver=PROJ_VER))
        run('unzip proj-datumgrid-1.5.zip -d proj-{projver}/nad/'.format(projver=PROJ_VER))
        with cd('proj-' + PROJ_VER):
            run('./configure --prefix={projdir} > log_proj_configure.out'.format(projdir=PROJ_DIR))
            run('make -j 4 > log_proj_make.out')
            sudo('make install > log_proj_make_install.out')
    sudo('sh -c "echo \'{projdir}/lib\' > /etc/ld.so.conf.d/proj.conf"'.format(projdir=PROJ_DIR))
    sudo('ldconfig')

@task
def install_geos():
    '''Install GEOS'''
    run('mkdir -p ' + GEOS_SRC)
    with cd(GEOS_SRC):
        run('wget http://download.osgeo.org/geos/geos-' + GEOS_VER + '.tar.bz2')
        run('tar xjf geos-' + GEOS_VER + '.tar.bz2')
        with cd('geos-' + GEOS_VER):
            run('./configure --prefix=' + GEOS_DIR + ' > log_geos_configure.out')
            run('make -j 4 > log_geos_make.out')
            sudo('make install > log_geos_make_install.out')
    sudo('sh -c "echo \'' + GEOS_DIR + '/lib\' > /etc/ld.so.conf.d/geos.conf"')
    sudo('ldconfig')


@task
def install_gdal():
    '''Install GDAL'''
    run('mkdir -p ' + GDAL_SRC)
    with cd(GDAL_SRC):
        run('wget http://download.osgeo.org/gdal/gdal-' + GDAL_VER + '.tar.gz')
        run('tar xzf gdal-' + GDAL_VER + '.tar.gz')
        with cd('gdal-' + GDAL_VER):
            run('./configure' + \
                ' --prefix=' + GDAL_DIR + \
                ' --with-geos=' + GEOS_DIR + '/bin/geos-config' + \
                ' --with-python' + \
                ' >& log_gdal_configure.out')
            run('make >& log_gdal_make.out')
            sudo('make install >& log_gdal_make_install.out')
    sudo('sh -c "echo \'' + GDAL_DIR + '/lib\' > /etc/ld.so.conf.d/gdal.conf"')
    sudo('ldconfig')


@task
def install_hdf5():
    '''Install HDF5'''
    sudo('apt-get install -y libcurl3 libcurl4-openssl-dev')
    run('mkdir -p ' + HDF5_SRC)
    with cd(HDF5_SRC):
        HDF5_TAR = 'hdf5-' + HDF5_VER + '.tar.gz'
        run('wget http://www.hdfgroup.org/ftp/HDF5/current/src/' + HDF5_TAR)
        run('tar -xzvf ' + HDF5_TAR)
        with cd('hdf5-' + HDF5_VER):
            run('./configure' + \
                ' --prefix=' + HDF5_DIR + \
                ' --enable-shared' + \
                ' --enable-hl' + \
                ' > log_hdf5_configure.log')
            run('make -j 4 > log_hdf5_make.log')
            sudo('make install >& log_hdf5_make_install.log')
    sudo('sh -c "echo \'' + HDF5_DIR + '/lib\' > /etc/ld.so.conf.d/hdf5.conf"')
    sudo('ldconfig')


@task
def install_netcdf4():
    '''Install NetCDF4'''
    run('mkdir -p ' + NETCDF4_SRC)
    with cd(NETCDF4_SRC):
        NETCDF4_TAR = 'netcdf-' + NETCDF4_VER + '.tar.gz'
        run('wget ftp://ftp.unidata.ucar.edu/pub/netcdf/' + NETCDF4_TAR)
        run('tar -xzvf ' + NETCDF4_TAR)
        with cd('netcdf-' + NETCDF4_VER):
            run('./configure' + \
                ' --prefix=' + NETCDF4_DIR + \
                ' --enable-netcdf-4' + \
                ' --with-hdf5=' + HDF5_DIR + \
                ' --enable-shared ' + \
                ' --enable-dap ' + \
                ' > log_netcdf4_configure.log')
            run('make ' + MAKEJOBS + '> log_netcdf4_make.log')
            sudo('make install >& log_netcdf4_make_install.log')
    sudo('sh -c "echo \'' + NETCDF4_DIR + '/lib\' > /etc/ld.so.conf.d/hdf5.conf"')
    sudo('ldconfig')


@task
def install_python_dependencies():
    '''Install required Python packages'''
    
    from virtualenv import VIRTUALENVDIR
    from virtualenv import VIRTUALENVNAME
    from virtualenv import VIRTUALENVWRAPPER_ACTIVATE
    from virtualenv import virtualenv
    
    sudo('apt-get -y install python-dev')
    sudo('apt-get -y install python-setuptools')
    sudo('apt-get -y install python-pip')
    sudo('pip install virtualenv')
    sudo('pip install virtualenvwrapper')
    
    with settings(warn_only=True):
        run('mkdir ' + VIRTUALENVDIR)
    # create the Python virtual environment
    with prefix(VIRTUALENVWRAPPER_ACTIVATE):
        run('mkvirtualenv --no-site-packages ' + VIRTUALENVNAME)
    
    # install symbolic link in the virtual environment to GDAL
    with settings(warn_only=True):
        run('ln -s ' + GDAL_DIR + '/bin/gdal-config ' + \
                '~/.virtualenvs/' + VIRTUALENVNAME + '/bin/gdal-config')
    
    with virtualenv():
        run('pip install yolk')
        run('pip install Django==1.3')
        run('pip install django-piston')
        run('pip install -e hg+https://bitbucket.org/tylere/django-piston#egg=piston')
        run('pip install numpy==1.5.1')
        run('pip install Shapely')
        run('pip install geojson')
        with prefix('export HDF5_DIR=' + HDF5_DIR):
            with prefix('export NETCDF4_DIR=' + NETCDF4_DIR):
                run('pip install netCDF4==0.9.4')
        # install the GDAL Python bindings
        run('pip install --no-install GDAL')
        # build package extensions 
        with cd('$HOME/.virtualenvs/' + VIRTUALENVNAME + '/build/GDAL'):
            run('python setup.py build_ext' + \
                ' --gdal-config=' + GDAL_DIR + '/bin/gdal-config' + \
                ' --library-dirs=' + GDAL_DIR + '/include')
        run('pip install --no-download GDAL')

@task
def install_pykml():
    '''Install pyKML and dependencies'''
    from virtualenv import virtualenv
    
    sudo('apt-get -y install libxml2')
    sudo('apt-get -y install libxslt1.1 libxslt-dev')    
    with virtualenv():
        run('pip install pykml')
