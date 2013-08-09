import ocgis


uri = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
variable = 'tasmax'
ocgis.env.DIR_DATA = '/data/maurer'

for output_format in ['csv+','shp','csv']:
    ops = ocgis.OcgOperations(dataset={'uri':uri,
                                 'variable':variable,
                                 'time_region':{'year':None,'month':[7]}},
                        output_format=output_format,prefix=output_format,
                        calc=[{'name': 'Frequency Duration', 'func': 'freq_duration', 'kwds': {'threshold': 25.0, 'operation': 'gte'}}],
                        calc_grouping=['month','year'],
                        geom='us_counties',select_ugid=[2778],aggregate=True,
                        calc_raw=False,spatial_operation='clip',
                        headers=['did', 'ugid', 'gid', 'year', 'month', 'day', 'variable', 'calc_name', 'value'],)
    ret = ops.execute()
    print ret