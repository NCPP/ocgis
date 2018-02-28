import os
from ConfigParser import SafeConfigParser

from fabric.operations import local, sudo
from fabric.state import env

conf_path = os.getenv('OCGIS_CONF_PATH')
parser = SafeConfigParser()
parser.read(conf_path)

env.user = parser.get('fabric', 'user')
env.hosts = [parser.get('fabric', 'hosts')]
env.key_filename = parser.get('fabric', 'key_filename')


def tfs(sequence):
    return ' '.join(sequence)


def fcmd(name, cmd):
    return name(tfs(cmd))


def fecho(msg):
    local('echo "{0}"'.format(msg), capture=True)


def set_rwx_permissions(path):
    # set the owner for the files to the environment user
    fcmd(sudo, ['chown', '-R', env.user, path])
    # set read, write, execute...
    fcmd(sudo, ['chmod', '-R', 'u=rwx', path])


def set_rx_permisions(path):
    # set the owner for the files to the environment user
    fcmd(sudo, ['chown', '-R', env.user, path])
    # set read, execute...
    fcmd(sudo, ['chmod', '-R', 'a-rwx', path])
    fcmd(sudo, ['chmod', '-R', 'ug=rwx', path])
