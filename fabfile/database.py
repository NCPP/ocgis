from fabric.api import task
from fabric.api import run, sudo, cd, prefix
from virtualenv import virtualenv
from tasks_system import SRCDIR, GEOS_DIR, PROJ_DIR

POSTGRESQL_VER = '8.4'
POSTGIS_VER = '1.5.2'
POSTGIS_SRC = SRCDIR + '/postgis/' + POSTGIS_VER
POSTGIS_DIR = '/usr/share/postgresql/' + POSTGRESQL_VER + '/contrib/postgis-1.5'
POSTGIS_TEMPLATE_DB = 'postgis-' + POSTGIS_VER + '-template'

# project database parameters
#DBUSER='ubuntu'
#DBOWNER = 'openclimategis_user'
DBNAME = 'openclimategis_sql'


@task
def install_postgresql():
    '''Install PostgreSQL'''
    sudo('apt-get install -y postgresql-' + POSTGRESQL_VER)
    sudo('apt-get install -y postgresql-server-dev-' + POSTGRESQL_VER)
    sudo('apt-get install -y libpq-dev')
    sudo('apt-get install -y postgresql-client-' + POSTGRESQL_VER)
    sudo('apt-get install -y libxml2-dev')

@task
def install_psycopg2():
    '''Install the Python PostgreSQL database adapter'''
    with virtualenv():
        run('pip install psycopg2==2.4')


@task
def drop_postgresql_user(username):
    # Drop a PostgreSQL user
    sudo(
        'dropuser {user}'.format(user=username),
        user='postgres'
    )

@task
def create_postgresql_user(username, password):
    # Create a PostgreSQL user for the django project
    cmd = ('CREATE USER {0}' + \
          ' WITH NOSUPERUSER CREATEDB NOCREATEROLE LOGIN' + \
          ' PASSWORD \'{passwd}\'' + \
          ';').format(username, passwd=password)
    sudo('psql -c "{cmd}";'.format(cmd=cmd), user='postgres')


@task
def install_postgis():
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
def drop_postgis_template_db():
    '''Drop the PostGIS template database'''
    sudo(
        'psql -c "UPDATE pg_database ' + \
                'SET datistemplate=\'false\' ' + \
                'WHERE datname=\'' + POSTGIS_TEMPLATE_DB + '\';"',
        user='postgres'
    )
    sudo('dropdb ' + POSTGIS_TEMPLATE_DB, user='postgres')


@task
def create_postgis_template_db():
    '''Create the PostGIS template database'''
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
def create_openclimategis_db(databasename, owner):
    '''Configure the PostGIS database'''
    sudo('createdb ' + databasename + ' -T ' + POSTGIS_TEMPLATE_DB, user='postgres')
    sudo(
        'psql -c "ALTER DATABASE ' + databasename + ' OWNER TO ' + owner + ';"', 
        user='postgres'
    )
