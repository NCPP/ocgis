from fabric.api import task, settings
from fabric.api import cd, run, sudo, put
from fabric.api import warn
from __init__ import env
from virtualenv import VIRTUALENVDIR
from virtualenv import VIRTUALENVNAME
from virtualenv import virtualenv


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
def syncdb():
    '''Synchronizes the Django database tables'''
    with cd('$HOME/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis'):
        with virtualenv():
            # suppress input prompting for a superuser
            run('./manage.py syncdb --noinput')


@task
def create_superuser():
    '''Creates a Django superuser that can log into the Django admin'''
    with cd('$HOME/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis'):
        with virtualenv():
            run('./manage.py createsuperuser')

@task
def register_archive_usgs_cida_maurer():
    '''Register the USGS CIDA Maurer et al. downscaled archive'''
    with cd('$HOME/.virtualenvs/openclimategis/src/' + \
                               'openclimategis/src/openclimategis'):
        with virtualenv():
            run('./manage.py register_archive' + \
                ' http://cida.usgs.gov/qa/thredds/dodsC/maurer/monthly')
