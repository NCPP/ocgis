from ConfigParser import SafeConfigParser
from fabric.contrib.project import rsync_project
from fabric.state import env
from fabric.decorators import task
from fabric.operations import sudo, run, put
from fabric.context_managers import cd
import os
from helpers import set_rwx_permissions, set_rx_permisions, fcmd, parser
import packages


@task
def deploy_test_local_copy(dest):
    from ocgis.test.base import TestBase

    dest = os.path.expanduser(dest)
    if not os.path.exists(dest):
        os.path.makedirs(dest)
    tdata = TestBase.get_tst_data()
    tdata.copy_files(dest, verbose=True)


@task()
def deploy_test_rsync():
    remote_dir = parser.get('server', 'dir_data')
    local_dir = os.getenv('OCGIS_DIR_TEST_DATA')

    # we want to create the local test directory on the remote server:
    #  http://docs.fabfile.org/en/latest/api/contrib/project.html#fabric.contrib.project.rsync_project
    assert not local_dir.endswith('/')

    # update permissions so files may be copied
    local_dir_name = os.path.split(local_dir)[1]
    test_data_path = os.path.join(remote_dir, local_dir_name)
    set_rwx_permissions(test_data_path)
    try:
        # synchronize the project
        rsync_project(remote_dir, local_dir=local_dir)
    finally:
        # remove write permissions on the files/directories
        set_rx_permisions(test_data_path)


@task
def ebs_mkfs():
    """Make a file system on a newly attached device."""

    cmd = ['mkfs', '-t', 'ext4', parser.get('aws', 'ebs_mount_name')]
    fcmd(sudo, cmd)


@task
def ebs_mount():
    """Mount an EBS volume."""

    cmd = ['mount', parser.get('aws', 'ebs_mount_name'), parser.get('server', 'dir_data')]
    fcmd(sudo, cmd)


@task
def list_storage():
    """List storage size of connected devices."""

    fcmd(run, ['lsblk'])

@task
def put_file(local_path, remote_path):
    put(local_path=local_path, remote_path=remote_path)


@task
def remove_dir(path, use_sudo='false'):
    """Remove the source directory."""

    cmd = ['rm', '-r', path]
    if use_sudo == 'true':
        fmeth = sudo
    elif use_sudo == 'false':
        fmeth = run
    else:
        raise NotImplementedError(use_sudo)

    fcmd(fmeth, cmd)


@task
def run_tests(target='all', branch='next', failed='false'):
    """
    Run unit tests on remote server.

    :param str target: The test target. Options are:
        * 'all' = Run all tests.
        * 'simple' = Run simple test suite.
    :param str branch: The target GitHub branch.
    :param str failed: If ``'true'``, run only failed tests.
    :raises: NotImplementedError
    """

    path = os.path.join(parser.get('server', 'dir_clone'), parser.get('git', 'name'))

    if target == 'simple':
        test_target = os.path.join(path, 'src', 'ocgis', 'test', 'test_simple')
    elif target == 'all':
        test_target = os.path.join(path, 'src', 'ocgis', 'test')
    else:
        raise NotImplementedError(target)

    with cd(path):
        fcmd(run, ['git', 'pull'])
        fcmd(run, ['git', 'checkout', branch])
        fcmd(run, ['git', 'pull'])

        cmd = ['nosetests', '-sv', '--with-id', test_target]
        if failed == 'true':
            cmd.insert(-1, '--failed')
        elif failed == 'false':
            pass
        else:
            raise NotImplementedError(failed)

        fcmd(run, cmd)


@task
def install_dependencies():
    # packages.NumpyInstaller('1.8.2').execute()

    hdf5 = packages.HDF5Installer('1.8.13')
    # hdf5.execute()

    netcdf4 = packages.NetCDF4Installer('4.3.2', hdf5)
    # netcdf4.execute()

    # packages.NetCDF4PythonInstaller('1.1.1', netcdf4).execute()

    geos = packages.GeosInstaller('3.4.2')
    # geos.execute()

    proj4 = packages.Proj4Installer('4.8.0', '1.5')
    # proj4.execute()

    gdal = packages.GDALInstaller('1.11.1', geos, proj4)
    # gdal.execute()

    # packages.CythonInstaller('0.21.1').execute()

    # packages.ShapelyInstaller('1.4.3', geos).execute()

    # packages.FionaInstaller('1.4.5', gdal).execute()

    # packages.RtreeInstaller('0.8.0').execute()

    # packages.CFUnitsInstaller('0.9.6').execute()

    esmf = packages.ESMFInstaller('6.3.0rp1')
    # esmf.execute()

    # packages.ESMPyInstaller(esmf).execute()

    packages.IcclimInstaller(branch_name='master').execute()


# @task
# def install_virtual_environment():
#     install_apt_package('python-dev')
#     install_apt_package('python-pip')
#     install_pip_package('virtualenv')
#     install_pip_package('virtualenvwrapper')
#
#     ## append environment information to profile file
#     lines = [
#     '',
#     '# Set the location where the virtual environments are stored',
#     'export WORKON_HOME=~/.virtualenvs',
#     '# Use the virtual environment wrapper scripts',
#     'source /usr/local/bin/virtualenvwrapper.sh',
#     '# Tell pip to only run if there is a virtualenv currently activated',
#     'export PIP_REQUIRE_VIRTUALENV=false',
#     '# Tell pip to automatically use the currently active virtualenv',
#     'export PIP_RESPECT_VIRTUALENV=true',
#     '# Tell pip to use virtual environment wrapper storage location',
#     'export PIP_VIRTUALENV_BASE=$WORKON_HOME',
#             ]
#     append('~/.profile', lines)
#     run(tfs(['source', '~/.profile']))
#
#     run(tfs(['mkvirtualenv', env.cfg['venv_name']]))
