import ocgis
from co_watersheds_subset import parse_narccap_filenames


def main():
    snippet = False
    ## city center coordinate
    geom = [-97.74278,30.26694]
    ocgis.env.DIR_OUTPUT = '/tmp/narccap'
    ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
    ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True
    ocgis.env.VERBOSE = True
    
    
    rds = parse_narccap_filenames(ocgis.env.DIR_DATA)
    calc = [{'func':'mean','name':'mean'},
            {'func':'median','name':'median'},
            {'func':'max','name':'max'},
            {'func':'min','name':'min'}]
    calc_grouping = ['month','year']
    ops = ocgis.OcgOperations(dataset=rds,calc=calc,calc_grouping=calc_grouping,
                              output_format='csv+',geom=geom,abstraction='point',
                              snippet=snippet)
    ret = ops.execute()