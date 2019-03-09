#!/usr/bin/env bash

# The base or working directory.
WD=~/scratch/

# Directory to write all the chunked files to.
CHUNKDIR=${WD}/chunking

# Path to the source field/grid netCDF file.
SRC=grid_global_19980_39960.nc

# Path to the destination field/grid netCDF file.
DST=grid_regional_700_1820.nc

# Chunking composition for the destination grid (n_chunks_y,n_chunks_x).
NCHUNKS_DST=10,10

# Path to the merged output weight file.
WEIGHTS=${WD}/weights.nc

# Path to the output spatial subset of the global grid.
SS_PATH=${OUTDIR}/spatial_subset.nc

# The ocli python file or executable target with the subcommand for chunked
# regrid weight generation.
CRWG="ocli chunked-rwg "

# An optional execution prefix.
#EXEC_PREFIX=""
EXEC_PREFIX="mpirun -n `nproc`"

###############################################################################

# Run a spatial subset of the source grid using the spatial extent of the
# destination grid.

${EXEC_PREFIX} ${CRWG} --source ${SRC} --destination ${DST} --spatial_subset \
 --no_genweights --spatial_subset_path ${SS_PATH}

###############################################################################

# Generate the weights using the spatially subset source field and destination
# field.

${EXEC_PREFIX} ${CRWG} --source ${SS_PATH} --destination ${DST} \
 --esmf_regrid_method CONSERVE --nchunks_dst ${NCHUNKS_DST} --wd ${CHUNKDIR} \
 --weight ${WEIGHTS} --persist
