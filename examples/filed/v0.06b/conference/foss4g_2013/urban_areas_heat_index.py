import ocgis

ocgis.env.OVERWRITE = True
ocgis.env.DIR_DATA = '/usr/local/climate_data/CanCM4'
ocgis.env.VERBOSE = True


def main():
    ## create request datasets
    tas = ocgis.RequestDataset(uri='tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                               variable='tasmax')
    rhs = ocgis.RequestDataset(uri='rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                               variable='rhsmax')

    ## other operations arguments
    calc = [{'func': 'heat_index', 'name': 'heat_index',
             'kwds': {'tas': 'tasmax', 'rhs': 'rhsmax', 'units': 'k'}}]
    calc_grouping = ['month', 'year']
    snippet = False
    select_ugid = [668, 705, 743, 597, 634, 783, 599, 785, 600, 748, 786, 675, 676, 603, 827, 679, 680, 792, 682, 719,
                   794, 610, 760, 797, 835, 688, 692, 693, 694, 658, 695, 698, 735, 775, 666]
    geom = 'urban_areas_2000'
    aggregate = True
    spatial_operation = 'clip'
    output_format = 'shp'
    dir_output = '/home/local/WX/ben.koziol/Dropbox/nesii/conference/FOSS4G_2013/figures/heat_index'
    prefix = 'minneapolis_heat_index'
    agg_selection = True

    ## construct operations
    ops = ocgis.OcgOperations(dataset=[tas, rhs], calc_grouping=calc_grouping,
                              snippet=snippet, geom=geom, select_ugid=select_ugid, aggregate=aggregate,
                              spatial_operation=spatial_operation, output_format=output_format,
                              calc=calc, dir_output=dir_output, prefix=prefix, agg_selection=agg_selection)

    ## return the data
    ret = ops.execute()


def overlay():
    dir_output = '/home/local/WX/ben.koziol/Dropbox/nesii/conference/FOSS4G_2013/figures/cmip_overlay'
    prefix = 'cmip_grid'
    ## write cmip grid
    tas = ocgis.RequestDataset(uri='tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                               variable='tasmax')
    ops = ocgis.OcgOperations(dataset=tas, snippet=True, output_format='shp',
                              dir_output=dir_output, prefix=prefix)
    ret = ops.execute()


if __name__ == '__main__':
    main()
#    overlay()
