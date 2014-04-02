from fabric.state import env
from fabric.decorators import task
from fabric.operations import sudo
from fabric.context_managers import cd


env.user = 'ubuntu'
env.hosts = ['ec2-54-186-255-159.us-west-2.compute.amazonaws.com']
env.key_filename = '/home/local/WX/ben.koziol/.ssh/ocgis-bwk.pem'

env.dir_src = '/usr/local/src'
env.ver_hdf5 = 'hdf5-1.8.12'
env.ver_netcdf4 = 'netcdf-4.3.1.1'
env.ver_netcdf4_python = 'v1.0.8rel'
env.ver_cfunits_python = 'cfunits-0.9.6'
env.ver_ocgis = 'v0.07.1b'


@task(default=True)
def deploy():
    upgrade()
    install_apt_packages()
    install_pip_python_libraries()
    install_hdf5()
    install_netcdf4()
    install_netcdf4_python()
    install_cfunits_python()
    install_icclim()
    install_ocgis()

@task(default=False)
def upgrade():
    sudo('apt-get update')
    sudo('apt-get upgrade -y')
    
@task
def install_hdf5():
    url = 'http://www.hdfgroup.org/ftp/HDF5/current/src/{0}.tar.gz'.format(env.ver_hdf5)
    tar = '{0}.tar.gz'.format(env.ver_hdf5)
    sudo('mkdir -p '+env.dir_src)
    with cd(env.dir_src):
        sudo('wget '+url)
        sudo('tar -xzvf '+tar)
        with cd(env.ver_hdf5):
            sudo('./configure --prefix=/usr/local --enable-shared --enable-hl')
            sudo('make')
            sudo('make install')
            
@task
def install_netcdf4():
    url = 'ftp://ftp.unidata.ucar.edu/pub/netcdf/{0}.tar.gz'.format(env.ver_netcdf4)
    tar = '{0}.tar.gz'.format(env.ver_netcdf4)
    sudo('mkdir -p '+env.dir_src)
    with cd(env.dir_src):
        sudo('wget '+url)
        sudo('tar -xzvf '+tar)
        with cd(env.ver_netcdf4):
            sudo('LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include')
            sudo('./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local')
            sudo('make')
            sudo('make install')
#            sudo('make check')
            
@task
def install_netcdf4_python():
    url = 'https://github.com/Unidata/netcdf4-python/archive/{0}.tar.gz'.format(env.ver_netcdf4_python)
    tar = '{0}.tar.gz'.format(env.ver_netcdf4_python)
    sudo('mkdir -p '+env.dir_src)
    with cd(env.dir_src):
        sudo('wget '+url)
        sudo('tar -xzvf '+tar)
        with cd('netcdf4-python-{0}'.format(env.ver_netcdf4_python[1:])):
            sudo('ldconfig')
            sudo('python setup.py install')
            
@task
def install_cfunits_python():
    sudo('apt-get install -y libudunits2-0')
    sudo('mkdir -p '+env.dir_src)
    url = 'https://cfunits-python.googlecode.com/files/{0}.tar.gz'.format(env.ver_cfunits_python)
    tar = '{0}.tar.gz'.format(env.ver_cfunits_python)
    with cd(env.dir_src):
        sudo('wget '+url)
        sudo('tar -xzvf '+tar)
        with cd(env.ver_cfunits_python):
            sudo('python setup.py install')
            sudo('cp -r cfunits/etc /usr/local/lib/python2.7/dist-packages/cfunits')
            
@task
def install_pip_python_libraries():
    sudo('pip install shapely')
    sudo('pip install fiona')
    sudo('pip install nose')
    
@task
def install_ocgis():
    sudo('mkdir -p '+env.dir_src)
    with cd(env.dir_src):
        sudo('git clone https://github.com/NCPP/ocgis.git')
        with cd('ocgis'):
            sudo('python setup.py install')
            
@task
def install_icclim():
    sudo('mkdir -p '+env.dir_src)
    with cd(env.dir_src):
        sudo('git clone https://github.com/tatarinova/icclim.git')
        with cd('icclim'):
            sudo('gcc -fPIC -g -c -Wall ./icclim/libC.c -o ./icclim/libC.o')
            sudo('gcc -shared -o ./icclim/libC.so ./icclim/libC.o')
            sudo('python setup.py install')

@task
def install_apt_packages():
    cmd = ['apt-get','-y','install','g++','libz-dev','curl','wget','python-dev',
           'python-pip','libgdal-dev','ipython','python-gdal','git']
    sudo(' '.join(cmd))
    