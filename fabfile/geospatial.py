import os
from fabric.decorators import task
import __init__ as fab
from collections import namedtuple


Meta = namedtuple('Meta',['ver','tar_name','url','src','install','base'])
def get_meta(name,ext='tar.gz'):
    ver = fab.cp.get(name,'ver')
    base = fab.cp.get(name,'base').format(ver)
    tar_name = '{0}.{1}'.format(base,ext)
    url = fab.cp.get(name,'url').format(tar_name)
    src = os.path.join(fab.cp.get('install','src'),'{0}/v{1}'.format(name,ver))
    install = os.path.join(fab.cp.get('install','install'),'{0}/v{1}'.format(name,ver))
    return(Meta(ver=ver,
                tar_name=tar_name,
                url=url,
                src=src,
                install=install,
                base=base))
            

@task
def install_hdf():
    meta = get_meta('hdf5')
    fab.run('mkdir -p '+meta.src)
    with fab.cd(meta.src):
        fab.run('wget '+meta.url)
        fab.run('tar xzvf '+meta.tar_name)
        with fab.cd(meta.base):
            fab.run('./configure --prefix={0} --enable-shared --enable-hl'.format(meta.install))
#            fab.run('make -j {0}'.format(fab.J))
#            fab.sudo('make install')
#            fab.sudo('ldconfig')

@task
def install_netCDF4():
    meta = get_meta('netCDF4')
    meta_hdf5 = get_meta('hdf5')
    fab.run('mkdir -p '+meta.src)
    with fab.cd(meta.src):
        fab.run('wget '+meta.url)
        fab.run('tar xzvf '+meta.tar_name)
        with fab.cd(meta.base):
            fab.run('export LDFLAGS=-L{0}/lib'.format(meta_hdf5.install))
            fab.run('export CPPFLAGS=-I{0}R/include'.format(meta_hdf5.install))
            fab.run('export LD_LIBRARY_PATH={0}/lib'.format(meta_hdf5.install))
            fab.run('./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix='+meta.install)
#            fab.run('make -j {0}'.format(fab.J))
#            fab.sudo('make install')
#            fab.sudo('ldconfig')
        
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()