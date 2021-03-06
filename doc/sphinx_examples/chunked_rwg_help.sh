$ ocli chunked-rwg --help

Usage: ocli chunked-rwg [OPTIONS]

  Generate regridding weights using a spatial decomposition.

Options:
  -s, --source PATH               Path to the source grid NetCDF file.
                                  [required]
  -d, --destination PATH          Path to the destination grid NetCDF file.
                                  [required]
  -n, --nchunks_dst TEXT          Single integer or sequence defining the
                                  chunking decomposition for the destination
                                  grid. Each value is the number of chunks
                                  along each decomposed axis. For unstructured
                                  grids, provide a single value (i.e. 100).
                                  For logically rectangular grids, two values
                                  are needed to describe the x and y
                                  decomposition (i.e. 10,20). Required if
                                  --genweights and not --spatial_subset.
  --merge / --no_merge            (default=merge) If --merge, merge weight
                                  file chunks into a global weight file.
  -w, --weight PATH               Path to the output global weight file.
                                  Required if --merge.
  --esmf_src_type TEXT            (default=GRIDSPEC) ESMF source grid type.
                                  Supports GRIDSPEC, UGRID, ESMFMESH, and
                                  SCRIP.
  --esmf_dst_type TEXT            (default=GRIDSPEC) ESMF destination grid
                                  type. Supports GRIDSPEC, UGRID, ESMFMESH,
                                  and SCRIP.
  --genweights / --no_genweights  (default=True) Generate weights using ESMF
                                  for each source and destination subset.
  --esmf_regrid_method TEXT       (default=CONSERVE) The ESMF regrid method.
                                  Only applicable with --genweights. Supports
                                  CONSERVE, BILINEAR. PATCH, and NEAREST_STOD.
  --spatial_subset / --no_spatial_subset
                                  (default=no_spatial_subset) Optionally
                                  subset the destination grid by the bounding
                                  box spatial extent of the source grid. This
                                  will not work in parallel if --genweights.
  --src_resolution FLOAT          Optionally overload the spatial resolution
                                  of the source grid. If provided, assumes an
                                  isomorphic structure. Spatial resolution is
                                  the mean distance between grid cell center
                                  coordinates.
  --dst_resolution FLOAT          Optionally overload the spatial resolution
                                  of the destination grid. If provided,
                                  assumes an isomorphic structure. Spatial
                                  resolution is the mean distance between grid
                                  cell center coordinates.
  --buffer_distance FLOAT         Optional spatial buffer distance (in units
                                  of the destination grid coordinates) to use
                                  when subsetting the source grid by the
                                  spatial extent of a destination grid or
                                  chunk. This is computed internally if not
                                  provided. Useful to override if the area of
                                  influence for a source-destination mapping
                                  is known a priori.
  --wd PATH                       Optional working directory for intermediate
                                  chunk files. Creates a directory in the
                                  system's temporary scratch space if not
                                  provided.
  --persist / --no_persist        (default=no_persist) If --persist, do not
                                  remove the working directory --wd following
                                  execution.
  --eager / --not_eager           (default=eager) If --eager, load all data
                                  from the grids into memory before
                                  subsetting. This will increase performance
                                  as loading data for each chunk is avoided.
                                  Set this to --not_eager for a more memory
                                  efficient execution at the expense of
                                  additional IO operations.
  --ignore_degenerate / --no_ignore_degenerate
                                  (default=no_ignore_degenerate) If
                                  --ignore_degenerate, skip degenerate
                                  coordinates when regridding and do not raise
                                  an exception.
  --data_variables TEXT           List of comma-separated data variable names
                                  to overload auto-discovery.
  --spatial_subset_path PATH      Optional path to the output spatial subset
                                  file. Only applicable when using
                                  --spatial_subset.
  --verbose / --not_verbose       If True, log to standard out using verbosity
                                  level.
  --loglvl TEXT                   Verbosity level for standard out logging.
                                  Default is "INFO". See Python logging level
                                  docs for additional values:
                                  https://docs.python.org/3/howto/logging.html
  --weightfilemode TEXT           The ESMF FileMode constant value. BASIC (the
                                  default) only writes the factor index list
                                  and weight factor variables. WITHAUX adds
                                  auxiliary variables and additional file
                                  metadata to the output weight file.
  --64bit_offset / --no_64bit_offset
                                  If provided, create the weight file in
                                  NetCDF using the 64-bit offset format to
                                  allow variables larger than 2GB. Note the
                                  64-bit offset format is not supported in the
                                  NetCDF version earlier than 3.6.0.  An error
                                  message will be generated if this flag is
                                  specified while the application is linked
                                  with a NetCDF library earlier than 3.6.0.
  --help                          Show this message and exit.