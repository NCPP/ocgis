from contextlib import contextmanager as _contextmanager
from fabric.api import task
from fabric.api import prefix
from fabric.api import run

# Python virtual environment parameters
VIRTUALENVDIR = '$HOME/.virtualenvs/'
VIRTUALENVNAME = 'openclimategis'
VIRTUALENVWRAPPER_ACTIVATE = 'source /usr/local/bin/virtualenvwrapper.sh'

@_contextmanager
def virtualenv():
    '''Execute command in a Python virtual environment'''
    #with cd(env.directory):
    with prefix(VIRTUALENVWRAPPER_ACTIVATE):
        with prefix('workon {venv}'.format(venv=VIRTUALENVNAME)):
            yield

@task
def list_virtualenv_packages():
    ''' List the virtual environment packages'''
    with virtualenv():
        run('yolk -l')
