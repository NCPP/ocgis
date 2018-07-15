"""
Use OCGIS grid chunking to yield a spatial chunk from source and destination grids. Spatial chunks are converted to
xarray datasets.
"""

import os
import tempfile

import dask
import ocgis
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.test import create_gridxy_global, create_exact_field
import numpy as np


OUTDIR = tempfile.gettempdir()
GRIDFN = os.path.join(OUTDIR, 'grid.nc')


# Create a test grid
grid = create_gridxy_global(crs=ocgis.crs.Spherical())
field = create_exact_field(grid, 'foo')
field.write(GRIDFN)


def apply_by_spatial_chunk(src_filename, dst_filename, nchunks, chunk_idx, **kwargs):
    """
    Create a spatial chunk from source and destination CF-Grid NetCDF files. Each source and destination chunk is
    converted to a :class:`xarray.Dataset`. See :class:`~ocgis.spatial.grid_chunker.GridChunker` for more documentation
    on the spatial chunking.

    Returns `0` if the chunking is successful.

    :param str src_filename: Path to source NetCDF file.
    :param str dst_filename: Path to destination NetCDF file.
    :param nchunks: The chunking decomposition for the destination grid. See :class:`~ocgis.spatial.grid_chunker.GridChunker`.
    :type nchunks: tuple(int, ...)
    :param int chunk_idx: The target chunk index.
    :param kwargs: Extra keyword arguments to :class:`~ocgis.spatial.grid_chunker.GridChunker` initialization.
    :rtype: int
    """
    rc = 1
    rd_src = ocgis.RequestDataset(src_filename)
    rd_dst = ocgis.RequestDataset(dst_filename)
    gc = GridChunker(rd_src, rd_dst, nchunks_dst=nchunks, **kwargs)
    for ctr, (src_grid, src_slice, dst_grid, dst_slice) in enumerate(gc.iter_src_grid_subsets(yield_dst=True, yield_idx=chunk_idx)):
        xsrc = src_grid.parent.to_xarray()
        xdst = dst_grid.parent.to_xarray()
        rc = 0
    assert ctr == 0  # Ensure we only have a single loop
    return rc


nchunks = (5, 5)  # The chunking decomposition for the destination grid. Five chunks along each spatial dimension.
results = []  # Will hold integer return codes (a placeholder for another return type)
for ii in range(np.prod(nchunks)):  # Each chunk is a separate dask task
    d = dask.delayed(apply_by_spatial_chunk)(GRIDFN, GRIDFN, nchunks, ii)  # Graph the chunking decomposition
    results.append(d)

dask.delayed(print)(results).compute(scheduler='threads')
