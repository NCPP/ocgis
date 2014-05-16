import ocgis


ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/presentation/20140514_ocgis_flyer'
SHP_PATH = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/shp/qed_tbw_basins/qed_tbw_basins.shp'
NC_PATH = '/home/local/WX/ben.koziol/climate_data/maurer/2010-concatenated/Maurer02new_OBS_pr_daily.1971-2000.nc'


rd = ocgis.RequestDataset(uri=NC_PATH, variable='pr')
calc = [{'func': 'freq_perc', 'name': 'threshold', 'kwds': {'percentile': 90}}]
calc_grouping = ['month']
ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, time_region={'month': [7], 'year': [1990]},
                          output_format='shp', geom=SHP_PATH, spatial_operation='clip', aggregate=True, prefix='tbw')
# print(ops.execute())

ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format='shp', prefix='maurer_grid')
print(ops.execute())
