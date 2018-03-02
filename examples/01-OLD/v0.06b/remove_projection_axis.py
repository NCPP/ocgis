import os

import netCDF4 as nc

import ocgis

## path to input file
NCFILE = '/home/local/WX/ben.koziol/climate_data/projection-axis/Canada_Projections.nc'
## directory to write projection files to
OUTDIR = '/tmp/foo'
## string template for making unique output names
OUT_TEMPLATE = 'Canada_Projections_Tavg_p{0}.nc'


def write_projection_file(ds, pidx):
    print('writing projection: {0}'.format(pidx + 1))
    ## path to the nc file holding a single projection
    out_path = os.path.join(OUTDIR, OUT_TEMPLATE.format(str(pidx + 1).zfill(2)))
    ## open the output dataset with the same format as the input
    ds_out = nc.Dataset(out_path, 'w', format=ds.file_format)
    try:
        ## transfer attributes to new datasets
        ds_out.setncatts(ds.__dict__)
        ## construct output dimensions - note the projection dimension is missing
        for dimname in ['time', 'latitude', 'longitude']:
            ds_out.createDimension(dimname, size=len(ds.dimensions[dimname]))
        ## create the coordinate variables
        for varname in ['time', 'latitude', 'longitude']:
            r_dvar = ds.variables[varname]
            var = ds_out.createVariable(varname, r_dvar.dtype, r_dvar.dimensions)
            ## remove bounds attribute as this does not exist in the dataset
            new_attrs = r_dvar.__dict__.copy()
            new_attrs.pop('bounds', None)
            var.setncatts(new_attrs)
            ## this is where the data is pulled and assigned to the new variable.
            ## the sync at the end is not really necessary for smaller data blocks
            ## but may be useful for larger files
            var[:] = r_dvar[:]
            ds_out.sync()
        ## create and fill the data variable using the projection index
        orig_var = ds.variables['Tavg']
        data_var = ds_out.createVariable('Tavg', orig_var.dtype, ('time', 'latitude', 'longitude'))
        data_var.setncatts(orig_var.__dict__)
        data_var[:] = orig_var[pidx, :, :, :]
    finally:
        ds_out.close()
    ## ensure data may be read by OCGIS
    ocgis.Inspect(out_path, variable='Tavg')


def main():
    ds = nc.Dataset(NCFILE, 'r')
    try:
        for pidx in range(len(ds.dimensions['projection'])):
            write_projection_file(ds, pidx)
    finally:
        ds.close()
    print('success.')


if __name__ == '__main__':
    main()
