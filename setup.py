from distutils.core import setup
import sys
import os


try:
    arg = sys.argv[1]
except IndexError:
    raise(ValueError('Please supply an "install" or "uninstall" command.'))

if arg == 'install':
    ## check python version
    version = float(sys.version_info.major) + float(sys.version_info.minor)/10
    if version < 2.7 or version > 2.7:
        raise(ImportError('This software requires Python version 2.7.'))
    
    ## attempt package imports
    pkgs = ['numpy','netCDF4','osgeo','shapely']
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            msg = 'Unable to import Python package: "{0}".'.format(pkg)
            raise(ImportError(msg))
    
    ## get package structure
    def _get_dot_(path,root='src'):
        ret = []
        path_parse = path
        while True:
            path_parse,tail = os.path.split(path_parse)
            if tail == root:
                break
            else:
                ret.append(tail)
        ret.reverse()
        return('.'.join(ret))
    package_dir = {'':'src'}
    src_path = os.path.join(package_dir.keys()[0],package_dir.values()[0],'ocgis')
    packages = []
    for dirpath,dirnames,filenames in os.walk(src_path):
        if '__init__.py' in filenames:
            package = _get_dot_(dirpath)
            packages.append(package)
    
    ## run the installation
    setup(name='ocgis',
          version='0.04.01b',
          author='Ben Koziol',
          author_email='ben.koziol@noaa.gov',
          url='https://github.com/NCPP/ocgis/tags',
          license='BSD License',
          platforms=['all'],
          packages=packages,
          package_dir=package_dir
          )
elif arg == 'uninstall':
    raise(NotImplementedError)
    try:
        first = True
        while True:
            try:
                __import__('ocgis')
                #TODO: the actual uninstall work
                first = False
            except ImportError:
                if first:
                    raise
                else:
                    break
    except ImportError:
        raise(ImportError("Either OpenClimateGIS is not installed or not available on the Python PATH."))
else:
    raise(ValueError('Only "install" and "uninstall" commands are supported. Command not recognized: "{0}".'.format(arg)))