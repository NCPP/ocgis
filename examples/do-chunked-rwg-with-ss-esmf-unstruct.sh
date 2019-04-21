#!/usr/bin/env bash

# Download URLs for the test files
URLSRC="https://www.dropbox.com/s/d7r1hv649vmwom5/ll1280x1280_grid.esmf.nc"
URLDST="https://www.dropbox.com/s/i2g7vv4gbto65k8/ll1280x1280_grid.esmf.subset.nc"

# The base or working directory.
WD=~/scratch/

# Directory to write all the chunked files to.
CHUNKDIR=${WD}/chunking

# Path to the source field/grid netCDF file.
SRC=ll1280x1280_grid.esmf.nc
SRCTYPE="ESMFMESH"
SRC_MAXSPATIALRES=0.28125

# Path to the destination field/grid netCDF file.
DST=ll1280x1280_grid.esmf.subset.nc
DSTTYPE="ESMFMESH"
DST_MAXSPATIALRES=0.28125

# Chunking composition for the destination grid (n_chunks_y,n_chunks_x).
NCHUNKS_DST=10

# Path to the merged output weight file.
WEIGHTS=${WD}/weights.nc

# Path to the output spatial subset of the global grid.
SS_PATH=${WD}/spatial_subset.nc

# The ocli python file or executable target with the subcommand for chunked
# regrid weight generation.
CRWG="ocli chunked-rwg "
#CRWG="python /home/ubuntu/Dropbox/NESII/project/ocg/git/ocgis/src/ocgis/ocli.py chunked-rwg "

# An optional execution prefix.
EXEC_PREFIX=""
#EXEC_PREFIX="mpirun -n `nproc`"

###############################################################################

# Download the example data files.

if [ ! -f ${SRC} ]; then
    wget ${URLSRC}
fi
if [ ! -f ${DST} ]; then
    wget ${URLDST}
fi

###############################################################################

# Run a spatial subset of the source grid using the spatial extent of the
# destination grid.

${EXEC_PREFIX} ${CRWG} --source ${SRC} --destination ${DST} --spatial_subset \
 --no_genweights --spatial_subset_path ${SS_PATH} --esmf_src_type ${SRCTYPE} \
 --esmf_dst_type ${DSTTYPE} --src_resolution ${SRC_MAXSPATIALRES} --verbose

###############################################################################

# Generate the weights using the spatially subset source field and destination
# field.

${EXEC_PREFIX} ${CRWG} --source ${SS_PATH} --destination ${DST} \
 --esmf_regrid_method CONSERVE --nchunks_dst ${NCHUNKS_DST} --wd ${CHUNKDIR} \
 --weight ${WEIGHTS} --persist --esmf_src_type ${SRCTYPE} --esmf_dst_type ${DSTTYPE} \
 --src_resolution ${SRC_MAXSPATIALRES} --dst_resolution ${DST_MAXSPATIALRES} --verbose
