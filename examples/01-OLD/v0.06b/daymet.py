import ocgis

uri = '/usr/local/climate_data/daymet/tmax.nc'
# uri = 'http://daymet.ornl.gov/thredds//dodsC/allcf/2011/9947_2011/tmax.nc'
variable = 'tmax'


def test():
    ocgis.env.OVERWRITE = True

    rd = ocgis.RequestDataset(uri=uri, variable=variable)
    ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format='nc')
    ops.execute()


def test_bounds():
    import netCDF4 as nc
    #    uri = 'http://daymet.ornl.gov/thredds//dodsC/allcf/2011/9947_2011/tmax.nc'
    uri = 'http://daymet.ornl.gov/thredds//dodsC/allcf/2011/11926_2011/tmax.nc'
    ds = nc.Dataset(uri, 'r')
    try:
        calendar = ds.variables['time'].calendar
        units = ds.variables['time'].units
        centroid = nc.num2date(ds.variables['time'][:], units, calendar=calendar)
        bounds = nc.num2date(ds.variables['time_bnds'][:], units, calendar=calendar)
        assert (centroid[0].year == bounds.flat[0].year)
    finally:
        ds.close()
