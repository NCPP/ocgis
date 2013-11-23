import ocgis
import numpy as np


ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/bcca/obs/tasmax/1_8deg'
ocgis.env.DIR_OUTPUT = '/tmp/nctest'
ocgis.env.OVERWRITE = True

print('getting geometry...')
rd = ocgis.RequestDataset(uri='gridded_obs.tasmax.OBS_125deg.daily.1999.nc',
                          variable='tasmax')
ops = ocgis.OcgOperations(dataset=rd,snippet=True)
ret = ops.execute()

print('getting selection calculation...')
ncgeoms = ret[1].variables['tasmax'].spatial._value
prefix_template = 'row_{0}'
for rowidx in range(ncgeoms.shape[0]):
    prefix = prefix_template.format(rowidx)
    print(prefix)
    row = ncgeoms[rowidx,:]
    geom = [None]*row.shape[0]
    for idx in range(len(geom)):
        geom[idx] = {'ugid':idx+1,'geom':row[idx]}
    #geom = [{'ugid':1,'geom':ret[1].variables['tasmax'].spatial.value.compressed()[0]}]
    calc = [{'func':'freq_perc','name':'perc_95','kwds':{'perc':0.95,'round_method':'ceil'}}]
    ops = ocgis.OcgOperations(dataset=rd,calc=calc,geom=geom,calc_grouping=['month'],
                              agg_selection=True,prefix=prefix,output_format='nc')
    retc = ops.execute()

print('success.')