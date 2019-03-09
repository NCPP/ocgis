$ ocli chunked-smm --help
Usage: ocli.py chunked-smm [OPTIONS]

  Apply weights in chunked files with an option to insert the global data.

Options:
  --wd PATH                       Optional working directory containing
                                  destination chunk files. If empty, the
                                  current working directory is used.
  --index_path FILE               Path grid chunker index file. If not
                                  provided, it will assume the default name in
                                  the working directory.
  --insert_weighted / --no_insert_weighted
                                  If --insert_weighted, insert the weighted
                                  data back into the global destination file.
  -d, --destination FILE          Path to the destination grid NetCDF file.
                                  Needed if using --insert_weighted.
  --data_variables TEXT           List of comma-separated data variable names
                                  to overload auto-discovery.
  --help                          Show this message and exit.
