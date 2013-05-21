import ocgis
from ocgis.calc import tile
import time
import cProfile
import pstats


def main():
    ocgis.env.OPTIMIZE_FOR_CALC = True
#    215 minutes
    uri = '/tmp/gridded_obs.tasmax.OBS_125deg.daily.1950-1999.nc'
#    uri = '/usr/local/climate_data/maurer/bcca/obs/tasmax/1_8deg/gridded_obs.tasmax.OBS_125deg.daily.1950.nc'
#        uri = '/usr/local/climate_data/daymet/tmax.nc'
    variable = 'tasmax'
#        variable = 'tmax'
    rd = ocgis.RequestDataset(uri,variable)
    import netCDF4 as nc
    ods = ocgis.api.dataset.dataset.OcgDataset(rd)
    shp = ods.i.spatial.shape
    print('getting schema...')
    schema = tile.get_tile_schema(shp[0],shp[1],100)
    calc = [{'func':'mean','name':'my_mean'},
            {'func':'freq_perc','name':'perc_90','kwds':{'perc':90,}},
            {'func':'freq_perc','name':'perc_95','kwds':{'perc':95,}},
            {'func':'freq_perc','name':'perc_99','kwds':{'perc':99,}}
           ]
    print('getting fill file...')
    fill_file = ocgis.OcgOperations(dataset=rd,file_only=True,
                                  calc=calc,calc_grouping=['month'],
                                  output_format='nc').execute()
    print fill_file, len(schema)
    fds = nc.Dataset(fill_file,'a')
    t1 = time.time()
    for tile_id,indices in schema.iteritems():
        print tile_id
        row = indices['row']
        col = indices['col']
        ret = ocgis.OcgOperations(dataset=rd,slice_row=row,slice_column=col,
                                  calc=calc,calc_grouping=['month'],
                                  abstraction='point').execute()
        ref = ret[1].variables[variable].calc_value
        for k,v in ref.iteritems():
            vref = fds.variables[k]
            if len(vref.shape) == 3:
                vref[:,row[0]:row[1],col[0]:col[1]] = v
            elif len(vref.shape) == 4:
                vref[:,:,row[0]:row[1],col[0]:col[1]] = v
            else:
                raise(NotImplementedError)
            fds.sync()
    fds.close()
    print((time.time()-t1)/60.0)
    
    
if __name__ == '__main__':
    cProfile.run('main()','/tmp/pstats')
    pp = pstats.Stats('/tmp/pstats')
    pp.strip_dirs().sort_stats('time').print_stats(20)