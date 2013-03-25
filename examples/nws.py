import ocgis


def main():
    ocgis.env.CORES = 2
    ocgis.env.SERIAL = True
    ocgis.env.DIR_DATA = '/usr/local/climate_data/CanCM4'
    
    
    ## build request datasets
    filenames = ['rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                 'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc']
    variables = ['rhsmax','tasmax']
    rds = [ocgis.RequestDataset(fn,var) for fn,var in zip(filenames,variables)]
    
    ## build calculations
    funcs = ['mean','std','min','max','median']
    calc = [{'func':func,'name':func} for func in funcs]
    
    ## operations
    select_ugid = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
    calc_grouping = ['month']
    snippet = True
    geom = 'climate_divisions'
    output_format = 'numpy'
    ops = ocgis.OcgOperations(dataset=rds,select_ugid=select_ugid,snippet=snippet,
     output_format=output_format,geom=geom,calc=calc,calc_grouping=calc_grouping)
    ret = ops.execute()
    import ipdb;ipdb.set_trace()
    
    
if __name__ == '__main__':
    main()