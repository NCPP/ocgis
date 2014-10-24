import abc
from helpers import fcmd
from fabfile import parser
import os
from fabric.context_managers import cd
from fabric.contrib.files import exists
from fabric.operations import sudo, run


class AbstractInstaller(object):
    __metaclass__ = abc.ABCMeta
    #: Set to a list of string package names to install using apt-get.
    apt_packages = None

    @abc.abstractproperty
    def prefix(self):
        """String prefix for the software."""

    def __init__(self, version):
        self.version = str(version)

    def execute(self):
        self.initialize()
        self.install()
        self.finalize()
        self.validate()

    def initialize(self):
        if self.apt_packages is not None:
            cmd = ['apt-get', '-y', 'install']
            cmd += self.apt_packages
            fcmd(sudo, cmd)

    @abc.abstractmethod
    def install(self):
        """Install the software."""

    @abc.abstractmethod
    def validate(self):
        """Command to validate the installation."""

    def finalize(self):
        """Any last steps with the installation."""

    def _get_filled_path_(self, base_dir):
        ret = os.path.join(base_dir, self.prefix, 'v{0}'.format(self.version))
        return ret


class AbstractMakeInstaller(AbstractInstaller):
    __metaclass__ = abc.ABCMeta
    dir_install = parser.get('server', 'dir_install')
    dir_src = parser.get('server', 'dir_src')
    configure_options = None
    make_check = True
    j = parser.get('server', 'j')

    @abc.abstractproperty
    def template_uncompressed_dir(self):
        """The folder name when the WGET file is decompressed."""

    @abc.abstractproperty
    def template_wget_url(self):
        """String with prefix and version used to get the WGET target URL."""

    @property
    def configure_cmd(self):
        cmd = ['./configure', '--prefix={0}'.format(self.path_install)]
        if self.configure_options is not None:
            cmd += self.configure_options
        return cmd

    @property
    def wget_url(self):
        return self.template_wget_url.format(version=self.version)

    @property
    def path_install(self):
        return self._get_filled_path_(self.dir_install)

    @property
    def path_src(self):
        return self._get_filled_path_(self.dir_src)

    @property
    def path_uncompressed_dir(self):
        return self.template_uncompressed_dir.format(version=self.version)

    @property
    def path_configure(self):
        return os.path.join(self.path_src, self.path_uncompressed_dir)

    def initialize(self):
        super(AbstractMakeInstaller, self).initialize()

        self._make_dir_src_()

        with cd(self.path_src):
            # download the source code
            cmd = ['wget', self.wget_url]
            fcmd(run, cmd)
            # uncompress
            fn = os.path.split(self.wget_url)[1]
            self.uncompress(fn)

    def configure(self):
        fcmd(run, self.configure_cmd)

    def install(self):
        with cd(self.path_configure):
            self.configure()
            cmd = ['make', '-j', self.j]
            fcmd(run, cmd)
            if self.make_check:
                fcmd(run, ['make', 'check'])
            fcmd(sudo, ['make', 'install'])

    @staticmethod
    def uncompress(fn):
        cmd = ['tar', '-xzvf', fn]
        fcmd(run, cmd)

    def validate(self):
        """Checking is implicit to ``install`` command."""

    def _make_dir_src_(self):
        # create the source directory. this should not exists to avoid duplicating installations.
        assert not exists(self.path_src)
        cmd = ['mkdir', '-p', self.path_src]
        fcmd(run, cmd)


class AbstractPythonInstaller(AbstractInstaller):
    __metaclass__ = abc.ABCMeta
    package_name = None

    def validate(self):
        cmd = '"import {0}"'.format(self.package_name or self.prefix)
        fcmd(run, ['python', '-c', cmd])


class AbstractPipInstaller(AbstractPythonInstaller):
    __metaclass__ = abc.ABCMeta

    def install(self):
        cmd = ['pip', 'install', '{0}=={1}'.format(self.prefix, self.version)]
        fcmd(sudo, cmd)


class AbstractSetupInstaller(AbstractPythonInstaller, AbstractMakeInstaller):
    __metaclass__ = abc.ABCMeta

    def execute(self):
        self.initialize()
        self.install()
        self.finalize()
        self.validate()

    def install(self):
        with cd(self.path_configure):
            fcmd(sudo, ['python', 'setup.py', 'install'])
