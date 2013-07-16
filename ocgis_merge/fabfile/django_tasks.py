import os, grp
from fabric.api import task, settings
from fabric.api import cd, run, sudo, put
from fabric.api import warn
from __init__ import env, get_settings_value
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
        with cd(VIRTUALENVDIR + VIRTUALENVNAME + '/src/openclimategis'):
            run('git pull')

@task
def copy_django_settings_config(localfile):
    '''Copy over a django settings configuration file
    
    Example: fab -H ubuntu@IPADDRESS django_tasks.copy_django_settings_config:localfile=/etc/openclimategis/settings.ini
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


#@task
#def create_log_file_directory():
#    '''Create log file directory for OpenClimateGIS'''
#    
#    settings_file='/etc/openclimategis/settings.ini'
#    
#    with virtualenv():
#        # get the log filename and path
#        logfile = get_settings_value(settings_file, 'logging', 'LOG_FILENAME')
#        # extract the directory
#        logfile_path,logfile_filename = os.path.split(logfile)
#        print(logfile_path)
#        logfile_path = '/home/terickson/temp/test-logging'
#        
#        # check if the directory exists
#        if not os.path.exists(logfile_path):
#            print('log file path does not exist!')
#            # if not, create it and assign group permissions to www-data
#            os.makedirs(logfile_path)
#            # change the group owner to www-data (used by apache)
#            gid = grp.getgrnam('www-data')
#            print gid
#            sudo os.chown(logfile_path, -1, gid.gr_gid)
#            #os.chmod()
#        else:
#            print('log file path exists!')
        
    

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
