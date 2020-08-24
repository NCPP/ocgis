import subprocess
import sys

import ocgis
from ocgis.vmachine.mpi import OcgDist


def main(nexe):
    dim = ocgis.Dimension(name="a", size=295335, dist=True)

    dist = OcgDist(size=nexe)
    dist.add_dimension(dim)
    dist.update_dimension_bounds()

    pipes = []
    for ii in range(nexe):
        local_bounds = dist.get_bounds_local(rank=ii)
        p = subprocess.Popen([sys.executable,
                              'convert_validate_esmf_mesh.py',
                              str(local_bounds[0][0]),
                              str(local_bounds[0][1])])
        pipes.append(p)
    for p in pipes:
        p.wait()


if __name__ == "__main__":
    main(int(sys.argv[1]))
