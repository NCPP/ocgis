import subprocess

from ocgis import env
from ocgis.test.base import create_gridxy_global
from ocgis.variable.crs import Spherical

SRC_PATH = '/home/benkoziol/htmp/rwg_test_src.nc'
DST_PATH = '/home/benkoziol/htmp/rwg_test_dst.nc'
WGT_PATH = '/home/benkoziol/htmp/rwg_test_weights.nc'
env.CLOBBER_UNITS_ON_BOUNDS = False

src_grid = create_gridxy_global(resolution=10.0, wrapped=False, crs=Spherical())
dst_grid = create_gridxy_global(resolution=20.0, wrapped=False, crs=Spherical())

src_grid.write(SRC_PATH)
dst_grid.write(DST_PATH)

cmd = ['ESMF_RegridWeightGen', '-s', SRC_PATH, '--src_type', 'GRIDSPEC', '-d', DST_PATH, '--dst_type', 'GRIDSPEC',
       '-w', WGT_PATH, '--method', 'conserve', '--weight-only']
subprocess.check_call(cmd)
