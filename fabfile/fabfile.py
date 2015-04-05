import os
import time

from fabric.contrib.project import rsync_project
from fabric.decorators import task
from fabric.operations import sudo, run, put, get
from fabric.context_managers import cd, shell_env, settings, prefix
from fabric.tasks import Task

from helpers import set_rwx_permissions, set_rx_permisions, fcmd, parser
from saws import AwsManager
from saws.tasks import ebs_mount


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
def list_storage():
    """List storage size of connected devices."""

    fcmd(run, ['lsblk'])

@task
def put_file(local_path, remote_path):
    """
    Put a file on the remote server: local_path,remote_path
    """

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


class RunAwsTests(Task):
    """
    Run tests on remote server and return the path to a local log file of tests results.
    """
    name = 'run_aws_tests'

    def run(self, path_local_log=None, branch='next', sched='false', launch_pause='false'):
        """
        :param str path_local_log: Path to the local log file copied from the remote server. If ``None``, do not copy
         remote log file.
        :param str branch: Target git branch to test.
        :param str sched: If ``'true'``, run tests only once. Otherwise, run tests at 23:00 hours daily.
        :param str launch_pause: If ``'true'``, pause at a breakpoint after launching the instance and mounting the data
         volume. Continuing from the breakpoint will terminate the instance and destroy the volume.
        """

        import schedule
        from logbook import Logger

        self.log = Logger('nesii-testing')

        self.path_local_log = path_local_log
        self.branch = branch
        self.launch_pause = launch_pause

        if self.launch_pause == 'true':
            self.log.info('launching instance then pausing')
            self._run_tests_(should_email=False)
        else:
            if sched == 'true':
                self.log.info('begin continous loop')
                schedule.every().day.at("6:00").do(self._run_tests_, should_email=True)
                while True:
                    schedule.run_pending()
                    time.sleep(1)
            else:
                self.log.info('running tests once')
                self._run_tests_(should_email=True)

    def _run_tests_(self, should_email=False):
        aws_src = os.getenv('OCGIS_SIMPLEAWS_SRC')
        aws_conf = os.getenv('OCGIS_CONF_PATH')
        aws_testing_section = 'aws-testing'

        ebs_volumesize = int(parser.get(aws_testing_section, 'ebs_volumesize'))
        ebs_snapshot = parser.get(aws_testing_section, 'ebs_snapshot')
        ebs_mount_name = parser.get(aws_testing_section, 'ebs_mount_name')
        ebs_placement = parser.get(aws_testing_section, 'ebs_placement')
        test_results_path = parser.get(aws_testing_section, 'test_results_path')
        test_instance_name = parser.get(aws_testing_section, 'test_instance_name')
        test_instance_type = parser.get(aws_testing_section, 'test_instance_type')
        test_image_id = parser.get(aws_testing_section, 'test_image_id')
        dest_email = parser.get(aws_testing_section, 'dest_email')
        dir_clone = parser.get('server', 'dir_clone')
        key_name = parser.get('simple-aws', 'key_name')

        import sys
        sys.path.append(aws_src)
        import saws
        import ipdb

        am = saws.AwsManager(aws_conf)

        self.log.info('launching instance')
        instance = am.launch_new_instance(test_instance_name, image_id=test_image_id, instance_type=test_instance_type,
                                          placement=ebs_placement)

        with settings(host_string=instance.ip_address, disable_known_hosts=True, connection_attempts=10):
            try:
                self.log.info('creating volume')
                volume = am.conn.create_volume(ebs_volumesize, ebs_placement, snapshot=ebs_snapshot)
                am.wait_for_status(volume, 'available')
                try:
                    self.log.info('attaching volume')
                    am.conn.attach_volume(volume.id, instance.id, ebs_mount_name, dry_run=False)
                    am.wait_for_status(volume, 'in-use')

                    ebs_mount()

                    if self.launch_pause == 'true':
                        self.log.info('pausing. continue to terminate instance...')
                        msg = 'ssh -i ~/.ssh/{0}.pem ubuntu@{1}'.format(key_name, instance.public_dns_name)
                        self.log.info(msg)
                        ipdb.set_trace()
                    else:
                        path = os.path.join(dir_clone, parser.get('git', 'name'))
                        test_target = os.path.join(path, 'src', 'ocgis', 'test')
                        # test_target = os.path.join(path, 'src', 'ocgis', 'test', 'test_simple')
                        nose_runner = os.path.join(path, 'fabfile', 'nose_runner.py')
                        path_src = os.path.join(path, 'src')
                        with cd(path):
                            fcmd(run, ['git', 'pull'])
                            fcmd(run, ['git', 'checkout', self.branch])
                            fcmd(run, ['git', 'pull'])
                        with cd(path_src):
                            with shell_env(OCGIS_TEST_TARGET=test_target):
                                fcmd(run, ['python', nose_runner])
                                if self.path_local_log is not None:
                                    get(test_results_path, local_path=self.path_local_log)

                    ebs_umount()

                finally:
                    self.log.info('detaching volume')
                    volume.detach(force=True)
                    am.wait_for_status(volume, 'available')
                    self.log.info('deleting volume')
                    volume.delete()
            finally:
                self.log.info('terminating instance')
                instance.terminate()

        if should_email and self.launch_pause == 'false' and self.path_local_log is not None:
            self.log.info('sending email')
            with open(self.path_local_log, 'r') as f:
                content = f.read()
            am.send_email(dest_email, dest_email, 'OCGIS_AWS', content)

        self.log.info('success')


r = RunAwsTests()


@task
def test_node_launch(run_tests='false'):
    am = AwsManager()
    instance_name = 'ocgis-test-node'
    image_id = 'ami-878aa5b7'
    instance_type = 't2.micro'
    ebs_snapshot_id = 'snap-310873bc'
    ebs_mount_dir = '~/data'
    ebs_mount_name = '/dev/xvdg'
    instance = am.launch_new_instance(instance_name, image_id=image_id, instance_type=instance_type,
                                      ebs_snapshot_id=ebs_snapshot_id, ebs_mount_name=ebs_mount_name)
    kwargs = {'mount_name': ebs_mount_name, 'mount_dir': ebs_mount_dir}
    am.do_task(ebs_mount, instance=instance, kwargs=kwargs)
    ssh_cmd = am.get_ssh_command(instance=instance)
    print ssh_cmd
    if run_tests == 'true':
        test_node_run_tests()
    print ssh_cmd


@task
def test_node_run_tests():
    am = AwsManager()
    instance_name = 'ocgis-test-node'
    instance = am.get_instance_by_name(instance_name)
    tbranch = 'next'
    tcenv = 'test_ocgis'
    texclude = '!slow,!remote,!esmpy7'
    tgdal_data = '/home/ubuntu/anaconda/envs/{0}/share/gdal'.format(tcenv)
    tocgis_dir_shpcabinet = '/home/ubuntu/data/ocgis_test_data/shp'
    tocgis_dir_test_data = '/home/ubuntu/data/ocgis_test_data/'
    tsrc = '~/git/ocgis/src'

    def _run_():
        senv = dict(OCGIS_DIR_SHPCABINET=tocgis_dir_shpcabinet, OCGIS_DIR_TEST_DATA=tocgis_dir_test_data,
                    GDAL_DATA=tgdal_data)
        with shell_env(**senv):
            with prefix('source activate {0}'.format(tcenv)):
                with cd(tsrc):
                    run('git pull')
                    cmd = 'cp .noseids /tmp; rm .noseids; git checkout {tbranch}; git pull; nosetests -vs --with-id -a {texclude} ocgis/test'
                    cmd = cmd.format(tbranch=tbranch, texclude=texclude)
                    run(cmd)

    am.do_task(_run_, instance=instance)


@task
def test_node_terminate():
    am = AwsManager()
    instance_name = 'ocgis-test-node'
    instance = am.get_instance_by_name(instance_name)
    instance.terminate()