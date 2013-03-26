import ocgis


def main():
    ocgis.env.CORES = 7
    ocgis.env.SERIAL = False
    ocgis.env.DIR_DATA = '/usr/local/climate_data/CanCM4'
    ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/presentation/2013_nws_gis_workshop'
    ocgis.env.OVERWRITE = True
    
    
    ## build request datasets
    filenames = [
#                 'rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                 'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc'
                 ]
    variables = [
#                 'rhsmax',
                 'tasmax'
                 ]
    rds = [ocgis.RequestDataset(fn,var) for fn,var in zip(filenames,variables)]
    
    ## build calculations
    funcs = ['mean','std']
#    funcs = ['mean','std','min','max','median']
    calc = [{'func':func,'name':func} for func in funcs]
    
    ## operations
#    select_ugid = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
    select_ugid = None
    calc_grouping = ['month','year']
    snippet = True
    geom = 'climate_divisions'
    output_format = 'shp'
    ops = ocgis.OcgOperations(dataset=rds,select_ugid=select_ugid,snippet=snippet,
     output_format=output_format,geom=geom,calc=calc,calc_grouping=calc_grouping,
     spatial_operation='clip',aggregate=True)
    ret = ops.execute()
    print(ret)
    
    
if __name__ == '__main__':
    main()