import ocgis


ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/links/project/ocg/presentation/2013-ESGF-F2F/ocgis_output'
ocgis.env.DIR_DATA = '/usr/local/climate_data'

rd_cmip5 = ocgis.RequestDataset('tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                                'tas',
                                alias='cmip5_tas')
rd_narccap = ocgis.RequestDataset('pr_CRCM_cgcm3_1981010103.nc',
                                  'pr',
                                  alias='narccap_pr')
rd_maurer = ocgis.RequestDataset('Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                                 'tasmax',
                                 alias='maurer_tasmax')

rds = [
    rd_cmip5,
    rd_narccap,
    rd_maurer
]

for rd in rds:
    print rd.alias
    ops = ocgis.OcgOperations(dataset=rd, output_format='shp', snippet=True, prefix=rd.alias,
                              output_crs=ocgis.crs.WGS84())
    ops.execute()

# #######################################################################################################################

for rd in rds:
    rd.time_region = {'month': [1]}
calc_grouping = ['month']
calc = [{'func': 'mean', 'name': 'mean'}, {'func': 'std', 'name': 'stdev'}]
ops = ocgis.OcgOperations(dataset=rds, geom='state_boundaries', select_ugid=[16], snippet=False, prefix='nebraska',
                          abstraction='point', aggregate=True, output_format='csv-shp', output_crs=ocgis.crs.WGS84(),
                          calc=calc, calc_grouping=calc_grouping, spatial_operation='clip')
ops.execute()