from fabric.api import task
from fabric.api import sudo

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
