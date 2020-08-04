import os
import sys

import numpy as np


def main(ncpath):
    import ESMF
    ESMF.Manager(debug=True)
    try:
        # Create the ESMF mesh
        mesh = ESMF.Mesh(filename=ncpath, filetype=ESMF.constants.FileFormat.ESMFMESH)
        # Create the source
        src = ESMF.Field(mesh, ndbounds=np.array([1, 1]), meshloc=ESMF.constants.MeshLoc.ELEMENT)
        # Create the destination
        dst = ESMF.Field(mesh, ndbounds=np.array([1, 1]), meshloc=ESMF.constants.MeshLoc.ELEMENT)
        # This will create the route handle and return some weights
        regrid = ESMF.Regrid(srcfield=src, dstfield=dst, regrid_method=ESMF.constants.RegridMethod.CONSERVE, factors=True)
        factors = regrid.get_weights_dict(deep_copy=True)
        assert factors is not None
    finally:
        if not os.path.exists('PET0.ESMF_LogFile'):
            sys.exit(1)
        with open('PET0.ESMF_LogFile', 'r') as f:
            lines = f.readlines()
        for l in lines:
            if 'ERROR' in l:
                sys.exit(1)


if __name__ == "__main__":
    ncpath = sys.argv[1]
    main(ncpath)
