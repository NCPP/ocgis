"""
Use chunked regrid weight generation and spatial subsetting for UGRID and SCRIP grid combinations.
"""

import os
import subprocess
import sys

import ocgis
from ocgis.util.logging_ocgis import ocgis_lh

DATA = {
    'ugrid': {'path': os.path.expanduser('~/l/i49-ugrid-cesm/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'),
              'nchunks_dst': 100,
              'etype': 'UGRID'},
    'scrip-struct': {'path': os.path.expanduser('~/l/i49-ugrid-cesm/0.9x1.25_c110307.nc'),
                     'nchunks_dst': 96,
                     'etype': 'SCRIP'},
    'scrip-unstruct': {'path': os.path.expanduser('~/l/i49-ugrid-cesm/SCRIPgrid_ne16np4_nomask_c110512.nc'),
                       'nchunks_dst': 50,
                       'etype': 'SCRIP'},
    'scrip-point': {'path': os.path.expanduser('~/l/i49-ugrid-cesm/SCRIPgrid_1x1pt_brazil_nomask_c110308.nc'),
                    'etype': 'SCRIP',
                    'is_point': True}}
BASEDIR = os.path.expanduser('~/htmp/cesm-manip')
WD = os.path.join(BASEDIR, 'chunks')
WEIGHT = os.path.join(BASEDIR, '01-global_weights.nc')
MPI_PROCS = 1
MPIEXEC = 'mpirun'
OCLI_EXE = os.path.expanduser('~/l/ocgis/src/ocli.py')
ocgis.env.VERBOSE = True

assert not os.path.exists(BASEDIR)
ocgis.env.configure_logging()


def create_command(wd, key_src, key_dst, weight, nprocs=MPI_PROCS):
    dsrc = DATA[key_src]
    ddst = DATA[key_dst]

    is_point = ddst.get('is_point', False)
    if is_point:
        nprocs = 1

    cmd = [MPIEXEC, '-n', str(nprocs), sys.executable, OCLI_EXE, 'chunked-rwg']

    cmd.extend(['--source', dsrc['path'], '--esmf_src_type', dsrc['etype']])
    cmd.extend(['--destination', ddst['path'], '--esmf_dst_type', ddst['etype']])
    cmd.extend(['--wd', wd])
    cmd.extend(['--weight', weight])
    if is_point:
        cmd.append('--spatial_subset')
    else:
        cmd.extend(['--nchunks_dst', str(ddst['nchunks_dst'])])
    # cmd.extend(['--no_genweights'])

    return cmd


if __name__ == '__main__':
    ocgis_lh(logger='chunker', msg='starting!')

    key_dst = 'scrip-unstruct'
    # key_dst = 'scrip-struct'
    # key_dst = 'scrip-point'
    cmd = create_command(WD, 'ugrid', key_dst, WEIGHT)

    ocgis_lh(logger='chunker', msg=' '.join(cmd))

    subprocess.check_call(cmd)

    ocgis_lh(logger='chunker', msg='stopping!')
