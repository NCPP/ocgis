from setuptools import setup, find_packages
import sys, os

version = '0.0.1'

setup(
    name='OpenClimateGIS',
    version=version,
    packages=['openclimategis',],
    package_dir={'': 'src'},
    package_data={
        'pykml': [
            'schemas/*.xsd',
            'test/*.py',
            'test/testfiles/*.kml',
            'test/testfiles/google_kml_developers_guide/*.kml',
            'test/testfiles/google_kml_tutorial/*.kml',
        ],
    },
    install_requires=[
        'setuptools',
        'django>=1.3',
    ],
    tests_require=['nose'],
    #test_suite='nose.collector',
    description="Python KML library",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Framework :: Django',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords='climate',
    author='Tyler Erickson',
    author_email='tylerickson@gmail.com',
    #url='http://pypi.python.org/pypi/openclimategis',
    license='BSD',
    long_description = open('README.rst').read(),
)
