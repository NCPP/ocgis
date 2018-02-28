import ocgis

ocgis.env.DIR_DATA = '/usr/local/climate_data'
ocgis.env.OVERWRITE = True

filenames = ['tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
             'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc']
variable = 'tasmax'
select_ugid = [32]  # Colorado

rd = ocgis.RequestDataset(uri=filenames, variable=variable)
ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=select_ugid,
                          output_format='csv+', snippet=False)
print(ops.execute())
