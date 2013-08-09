import ocgis
import itertools

#ocgis.env.DIR_BIN = '/home/local/WX/ben.koziol/links/ocgis/bin/QED_2013_dynamic_percentiles'

ocgis.env.DIR_DATA = '/data/maurer'

percentiles = [90,92.5,95,97.5]
operations = [
#              'gt',
              'gte',
#              'lt',
#              'lte'
              ]
calc_groupings = [
                  ['month'],
#                          ['month','year'],
#                          ['year']
                  ]
uris_variables = [['Maurer02new_OBS_tasmax_daily.1971-2000.nc','tasmax'],
                  ['Maurer02new_OBS_tasmin_daily.1971-2000.nc','tasmin']]
geoms_select_ugids = [
                      ['qed_city_centroids',None],
#                      ['state_boundaries',[39]],
#                              ['us_counties',[2416,1335]]
                      ]
for tup in itertools.product(percentiles,operations,calc_groupings,uris_variables,geoms_select_ugids):
    print(tup)
    percentile,operation,calc_grouping,uri_variable,geom_select_ugid = tup
    ops = ocgis.OcgOperations(dataset={'uri':uri_variable[0],'variable':uri_variable[1],'time_region':{'year':[1995,1996]}},
        geom=geom_select_ugid[0],select_ugid=geom_select_ugid[1],
        calc=[{'func':'qed_dynamic_percentile','kwds':{'operation':operation,'percentile':percentile},'name':'dp'}],
        calc_grouping=calc_grouping,output_format='numpy')
    ret = ops.execute()