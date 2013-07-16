from fabric.api import task
from fabric.api import sudo, cd

@task(default=True)
def install():
    '''Install the Apache HTTP Server'''
    sudo('apt-get install -y apache2 libapache2-mod-wsgi')

@task
def reload():
    '''Reloads the Apache2 Server configuration'''
    sudo('service apache2 reload')

@task
def config_openclimategis():
    '''Copy apache config file'''
    sudo('cp $HOME/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis/apache/vhost-config /etc/apache2/sites-available/openclimategis')
    sudo('a2ensite openclimategis')
    reload()

#@task
#def create_apache_static_folder():
#    '''Create a folder for hosting static files on the Apache HTTP server'''
#    sudo('mkdir /var/www/static')

@task
def make_local_copy_of_extjs():
    '''Make a local copy of the ExtJS libraries'''
    #sudo('mkdir -p /var/www/static/extjs')
    #with cd('/var/www/static/extjs'):
        #sudo('wget http://cdn.sencha.io/ext-4.0.7-gpl.zip')
        #sudo('unzip ext-4.0.7-gpl.zip')
        #sudo('mv ext-4.0.7-gpl 4.0.7')
    reload()