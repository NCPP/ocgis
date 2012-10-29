from fabric.decorators import task
from fabric.operations import run, sudo, local
from ConfigParser import ConfigParser
import geospatial
from fabric.context_managers import lcd

cp = ConfigParser()
cp.read('ocgis.cfg')
if cp.get('install','location') == 'local':
    run = local
    cd = lcd
    def lsudo(op):
        local('sudo {0}'.format(op))
    sudo = lsudo

SRC = cp.get('install','src')
INSTALL = cp.get('install','install')
J = cp.get('install','j')


@task(default=True)
def deploy():
#    geospatial.install_hdf()
    geospatial.install_netCDF4()