#!/usr/bin/env bash

# The base or working directory.
WD=~/scratch

# Directory to write all the chunked files to.
CHUNKDIR=${WD}/chunking

# Path to the source field/grid netCDF file.
SRC="/home/ubuntu/Dropbox/dtmp/ctsm-grids/SCRIPgrid_0.5x0.5_AVHRR_c110228.nc"
SRCTYPE="SCRIP"
SRC_MAXSPATIALRES=0.5

# Path to the destination field/grid netCDF file.
DST="/home/ubuntu/Dropbox/dtmp/ctsm-grids/SCRIPgrid_4x5_nomask_c110308.nc"
DSTTYPE="SCRIP"
DST_MAXSPATIALRES=0.5

# Chunking composition for the destination grid (n_chunks_y,n_chunks_x).
NCHUNKS_DST=10

# Path to the merged output weight file.
WEIGHTS=${WD}/weights.nc

# The ocli python file or executable target with the subcommand for chunked
# regrid weight generation.
CRWG="ocli chunked-rwg "
#CRWG="python /home/ubuntu/Dropbox/NESII/project/ocg/git/ocgis/src/ocgis/ocli.py chunked-rwg "

# An optional execution prefix.
#EXEC_PREFIX=""
EXEC_PREFIX="mpirun -n `nproc`"

###############################################################################

# Generate the weights using the spatially subset source field and destination
# field.

${EXEC_PREFIX} ${CRWG} --source ${SRC} --destination ${DST} \
 --esmf_regrid_method CONSERVE --nchunks_dst ${NCHUNKS_DST} --wd ${CHUNKDIR} \
 --weight ${WEIGHTS} --persist --esmf_src_type ${SRCTYPE} --esmf_dst_type ${DSTTYPE} \
 --src_resolution ${SRC_MAXSPATIALRES} --dst_resolution ${DST_MAXSPATIALRES} --verbose
