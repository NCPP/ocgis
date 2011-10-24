from fabric.api import task, local, env, cd, prefix
from fabric.api import run, sudo, put, warn
from fabric.api import settings
from fabric.tasks import Task
#from fabric.contrib import django
from contextlib import contextmanager as _contextmanager
from time import sleep
from boto.ec2.connection import EC2Connection

# Amazon Web Services parameters
aws_elastic_ip = '107.22.251.99'
USER = 'ubuntu'

# Python virtual environment parameters
VIRTUALENVDIR = '$HOME/.virtualenvs/'
VIRTUALENVNAME = 'openclimategis'
virtualenvwrapper_activate = 'source /usr/local/bin/virtualenvwrapper.sh'


# Fabric environment parameters
#env.user = 'ubuntu'
#env.hosts = [
#    'localhost',
#    '{user}@{host}'.format(user='ubuntu',host=aws_elastic_ip), # AWS Instance
#]
env.key_filename = '/home/terickson/.ssh/aws_openclimategis/ec2-keypair.pem'

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

POSTGRESQL_VER = '8.4'
POSTGIS_VER = '1.5.2'
POSTGIS_SRC = SRCDIR + '/postgis/' + POSTGIS_VER
POSTGIS_DIR = '/usr/share/postgresql/' + POSTGRESQL_VER + '/contrib/postgis-1.5'
POSTGIS_TEMPLATE_DB = 'postgis-' + POSTGIS_VER + '-template'

# project database parameters
#DBUSER='ubuntu'
DBOWNER = 'openclimategis_user'
DBNAME = 'openclimategis_sql'



@_contextmanager
def virtualenv():
    '''Execute command in a Python virtual environment'''
    #with cd(env.directory):
    with prefix(virtualenvwrapper_activate):
        with prefix('workon {venv}'.format(venv=VIRTUALENVNAME)):
            yield

@task
def create_aws_instance():
    '''Initialize an AWS instance'''
    
    print('Creating an AWS instance...')
    conn = EC2Connection()
    # start an instance of Ubuntu 10.04
    ami_ubuntu10_04 = conn.get_all_images(image_ids=['ami-3202f25b'])
    reservation = ami_ubuntu10_04[0].run( \
        key_name='ec2-keypair', \
        security_groups=['OCG_group'], \
        instance_type='m1.large', \
    )
    instance = reservation.instances[0]
    sleep(1)
    while instance.state!=u'running':
        print("Instance state = {0}".format(instance.state))
        instance.update()
        sleep(5)
    print("Instance state = {0}".format(instance.state))
    sleep(5)
    
    # add a tag to name the instance
    instance.add_tag('Name','OpenClimateGIS')
    
    print("DNS={0}".format(instance.dns_name))
    
    if conn.associate_address(instance.id, aws_elastic_ip):
        print('Success. Instance is now available at:{0}'.format(aws_elastic_ip))
    else:
        print('Failed. Unable to associate instance with public IP.')
    
    return instance.dns_name

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
    sudo('apt-get -y install swig')
    sudo('apt-get install -y git-core')

@task
def install_postgresql():
    '''Install PostgreSQL'''
    sudo('apt-get install -y postgresql-' + POSTGRESQL_VER)
    sudo('apt-get install -y postgresql-server-dev-' + POSTGRESQL_VER)
    sudo('apt-get install -y libpq-dev')
    sudo('apt-get install -y postgresql-client-' + POSTGRESQL_VER)
    sudo('apt-get install -y libxml2-dev')
    # Create a PostgreSQL user for the django project
    sudo(
        'createuser ' + DBOWNER + \
                        ' --no-superuser' + \
                        ' --createdb' + \
                        ' --no-createrole' + \
                        ' --pwprompt',
        user='postgres'
    )

@task
def create_source_code_folder():
    '''Creates a source code folder'''
    run('mkdir {0}'.format(srcdir))

@task
def install_geo_dependencies():
    '''Installs required geospatial libraries'''
    install_proj()
    install_geos()
    install_gdal()
    install_hdf5()
    install_netcdf4()

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
            run('./configure ' + \
                ' --prefix=' + GDAL_DIR + \
                ' --with-geos=' + GEOS_DIR + '/bin/geos-config' + \
                ' --with-python' + \
                ' >& log_gdal_configure.out')
            run('make -j 4 >& log_gdal_make.out')
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
def install_database():
    '''Install what the PostGIS database'''
    run('mkdir -p ' + POSTGIS_SRC)
    with cd(POSTGIS_SRC):
        run('wget http://postgis.refractions.net/' + \
            'download/postgis-' + POSTGIS_VER + '.tar.gz')
        run('tar xzf postgis-' + POSTGIS_VER + '.tar.gz')
        with cd('postgis-' + POSTGIS_VER):
            run('./configure' + \
                ' --with-geosconfig=' + GEOS_DIR + '/bin/geos-config' + \
                ' --with-projdir=' + PROJ_DIR + \
                ' > log_postgis_configure.out')
            run('make >& log_postgis_make.out')
            sudo('make install >& log_postgis_make_install.out')

@task
def create_template_database():
    '''Creates a PostGIS template database'''
    
    # drop the database
    sudo(
        'psql -c "UPDATE pg_database ' + \
                'SET datistemplate=\'false\' ' + \
                'WHERE datname=\'' + POSTGIS_TEMPLATE_DB + '\';"',
        user='postgres'
    )
    sudo('dropdb ' + POSTGIS_TEMPLATE_DB, user='postgres')
    
    # create a PostgreSQL database
    sudo('createdb ' + POSTGIS_TEMPLATE_DB, user='postgres')
    sudo('createlang plpgsql ' + POSTGIS_TEMPLATE_DB, user='postgres')
    # mark the database as a 'template' database
    sudo(
        'psql -c "UPDATE pg_database ' + \
                 'SET datistemplate=\'true\' ' + \
                 'WHERE datname=\'' + POSTGIS_TEMPLATE_DB + '\';"', 
        user='postgres'
    )
    # install the PostGIS functions
    sudo(
        'psql -d ' + POSTGIS_TEMPLATE_DB + \
            ' -f /usr/share/postgresql/' + POSTGRESQL_VER + \
                    '/contrib/postgis-1.5/postgis.sql', 
        user='postgres'
    )
    sudo(
        'psql -d ' + POSTGIS_TEMPLATE_DB + \
            ' -f /usr/share/postgresql/' + POSTGRESQL_VER + \
                    '/contrib/postgis-1.5/spatial_ref_sys.sql',
        user='postgres'
    )
    # grant access to the PostGIS geometry columns table
    sudo(
        'psql -d ' + POSTGIS_TEMPLATE_DB + \
            ' -c "GRANT ALL ON geometry_columns TO PUBLIC;"',
        user='postgres'
    )
    sudo(
        'psql -d ' + POSTGIS_TEMPLATE_DB + \
        ' -c "GRANT SELECT ON spatial_ref_sys TO PUBLIC;"',
        user='postgres'
    )

@task
def install_python_dependencies():
    '''Install required Python packages'''
    sudo('apt-get -y install python-dev')
    sudo('apt-get -y install python-setuptools')
    sudo('apt-get -y install python-pip')
    sudo('pip install virtualenv')
    sudo('pip install virtualenvwrapper')
    
    with settings(warn_only=True):
        run('mkdir ' + VIRTUALENVDIR)
    # create the Python virtual environment
    with prefix(virtualenvwrapper_activate):
        run('mkvirtualenv --no-site-packages ' + VIRTUALENVNAME)
    
    # install symbolic link in the virtual environment to GDAL
    with settings(warn_only=True):
        run('ln -s ' + GDAL_DIR + '/bin/gdal-config ' + \
                '~/.virtualenvs/' + VIRTUALENVNAME + '/bin/gdal-config')
    
    with virtualenv():
        run('pip install yolk')
        run('pip install Django==1.3')
        run('pip install django-piston')
        run('pip install psycopg2==2.4')
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
def list_virtualenv_packages():
    ''' List the virtual environment packages'''
    with virtualenv():
        run('yolk -l')

@task
def configure_openclimategis_database():
    '''Configure the PostGIS database'''
    sudo('createdb ' + DBNAME + ' -T ' + POSTGIS_TEMPLATE_DB, user='postgres')
    sudo(
        'psql -c "ALTER DATABASE ' + DBNAME + ' OWNER TO ' + DBOWNER + ';"', 
        user='postgres'
    )

@task
def install_openclimategis_django():
    '''Install OpenClimateGIS GeoDjango project'''
    with virtualenv():
        run('pip install -e git+http://github.com/tylere/OpenClimateGIS#egg=openclimategis')


@task
def update_openclimategis_django():
    '''Update the OpenClimateGIS GeoDjango project'''
    with virtualenv():
        #with cd('$HOME/.virtualenvs/openclimategis/src/openclimategis'):
        with cd(VIRTUALENVDIR + VIRTUALENVNAME + '/src/openclimategis'):
            run('git pull')


@task
def sync_django_database():
    with cd('$HOME/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis'):
        with virtualenv():
            run('./manage.py syncdb')

@task
def copy_django_settings_config(localfile):
    '''Copy over a django settings configuration file
    
    Example: fab -H ubuntu@IPADDRESS copy_django_settings_config:localfile=/etc/openclimategis/settings.ini
    '''
    if env.host=='localhost':
        warn('settings file not recopied to localhost')
    else:
        with settings(warn_only=True):
            sudo('mkdir /etc/openclimategis')
        remote_path = '/etc/openclimategis/settings.ini'
        put(
            local_path=localfile,
            remote_path=remote_path,
            use_sudo=True,
            mirror_local_mode=True,
        )
        # allow apache2 group ownership of the settings file
        sudo('chgrp www-data {0}'.format(remote_path))
        # make accessible to only the owner and group 
        sudo('chmod u+rw,g+r-w,o-rwx {0}'.format(remote_path))

@task
def update_apache_config():
    '''Copy apache config file'''
    sudo('cp $HOME/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis/apache/vhost-config /etc/apache2/sites-available/openclimategis')
    sudo('a2ensite openclimategis')
    sudo('/etc/init.d/apache2 reload')

@task
def register_archive_usgs_cida_maurer():
    '''Register the USGS CIDA Maurer et al. downscaled archive'''
    with cd('$HOME/.virtualenvs/openclimategis/src/' + \
                               'openclimategis/src/openclimategis'):
        with virtualenv():
            run('./manage.py register_archive' + \
                ' http://cida.usgs.gov/qa/thredds/dodsC/maurer/monthly')

@task
def apache2_reload():
    '''Reloads the Apache2 Server configuration'''
    sudo('service apache2 reload')

@task
def install_apache():
    '''Install the Apache HTTP Server'''
    sudo('apt-get install -y apache2 libapache2-mod-wsgi')


@task
def commit():
    '''Commit changes to version control'''
    local("git add -p && git commit")

@task
def prepare_deploy():
    test()
    #commit()

#@task
#def deploy():
#    code_dir = '/srv/django/myproject'
#    with cd(code_dir):
#        run("git pull")
#        run("touch app.wsgi")
