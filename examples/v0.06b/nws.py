import ocgis


def main():
    ocgis.env.DIR_DATA = '/usr/local/climate_data/CanCM4'
    ocgis.env.DIR_OUTPUT = '.../presentation/2013_nws_gis_workshop'
    
    rd = ocgis.RequestDataset(uri='tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                              variable='tasmax')
    calc = [{'func':'mean','name':'mean','func':'std','name':'std'}]
    ops = ocgis.OcgOperations(dataset=rd,geom='climate_divisions',spatial_operation='clip',
                              aggregate=True,calc=calc,calc_grouping=['month'],output_format='csv')
    ret = ops.execute()
    
    ## operations
#    select_ugid = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
    select_ugid = None
    calc_grouping = ['month']
    snippet = False
    geom = 'climate_divisions'
    output_format = 'csv'
    ops = ocgis.OcgOperations(dataset=rds,select_ugid=select_ugid,snippet=snippet,
     output_format=output_format,geom=geom,calc=calc,calc_grouping=calc_grouping,
     spatial_operation='clip',aggregate=True)
    ret = ops.execute()
    print(ret)
    
def write_climate_divisions():
    sc = ocgis.ShpCabinet()
    path = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/presentation/2013_nws_gis_workshop/climate_divisions/cd.shp'
    geoms = sc.get_geoms('climate_divisions')
    sc.write(geoms,path)
    
def write_overlay():
    ocgis.env.DIR_DATA = '/usr/local/climate_data/CanCM4'
    ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/presentation/2013_nws_gis_workshop'
    
    rd = ocgis.RequestDataset(uri='tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                              variable='tasmax')
    ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='shpidx',
                              prefix='overlay')
    ret = ops.execute()
    print(ret)
    
    
if __name__ == '__main__':
#    main()
#    write_climate_divisions()
    write_overlay()