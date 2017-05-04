import os
import sys

from setuptools import setup, Command, find_packages
from setuptools.command.test import test as TestCommand

VERSION = '1.3.2'


########################################################################################################################
# commands
########################################################################################################################


class test(TestCommand):
    user_options = [('with-optional', None, 'If present, run optional dependency tests.'),
                    ('no-esmf', None, 'If present, do not run ESMF tests.'),
                    ('no-icclim', None, 'If present, do not run ICCLIM tests.')]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.with_optional = False
        self.no_esmf = False
        self.no_icclim = False

    # def finalize_options(self):
    #     pass

    def run_tests(self):
        from ocgis.test import run_simple

        attrs = ['simple']
        if self.with_optional:
            to_append = 'optional'
            if self.no_esmf:
                to_append += ',!esmf'
            if self.no_esmf:
                to_append += ',!icclim'
            attrs.append(to_append)

        run_simple(attrs=attrs, verbose=False)


class test_all(TestCommand):
    def run_tests(self):
        from ocgis.test import run_all

        run_all(verbose=False)


class UninstallCommand(Command):
    description = "information on how to uninstall OCGIS"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import ocgis

            print('To uninstall, manually remove the Python package folder located here: {0}'.format(
                os.path.split(ocgis.__file__)[0]))
        except ImportError:
            raise (ImportError("Either OpenClimateGIS is not installed or not available on the Python path."))


########################################################################################################################
# check python version
########################################################################################################################


python_version = float(sys.version_info[0]) + float(sys.version_info[1]) / 10
if python_version != 2.7:
    raise (ImportError('This software requires Python version 2.7.x. You have {0}.x'.format(python_version)))


########################################################################################################################
# set up data files for installation
########################################################################################################################

shp_parts = ['state_boundaries.cfg', 'state_boundaries.dbf', 'state_boundaries.prj', 'state_boundaries.shp',
             'state_boundaries.shx']
shp_parts = ['bin/shp/state_boundaries/{0}'.format(element) for element in shp_parts]
bin_files = ['bin/test_csv_calc_conversion_two_calculations.csv']
bin_files += shp_parts
package_data = {'ocgis.test': bin_files}

########################################################################################################################
# setup command
########################################################################################################################

setup(
    name='ocgis',
    version=VERSION,
    author='NESII/CIRES/NOAA-ESRL',
    author_email='ocgis_support@list.woc.noaa.gov',
    url='http://ncpp.github.io/ocgis/install.html#installing-openclimategis',
    license='NCSA License',
    platforms=['all'],
    packages=find_packages(where='./src'),
    package_dir={'': 'src'},
    package_data=package_data,
    cmdclass={'uninstall': UninstallCommand,
              'test': test,
              'test_all': test_all},
    install_requires=['numpy', 'netCDF4', 'fiona', 'shapely'],
    tests_require=['nose']
)
