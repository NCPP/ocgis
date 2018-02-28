from ocgis.interface.base.crs import CFWGS84

import ocgis
from helpers import parse_narccap_filenames


def main():
    ## city center coordinate (~Austin, TX)
    geom = [-97.74278, 30.26694]
    ## directory to write output data. needs to exist!
    ocgis.env.DIR_OUTPUT = '/tmp/narccap'
    ## location of directory containing climate data files
    ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
    ## print additional information to console
    ocgis.env.VERBOSE = True

    ## load the request datasets by parsing filenames on disk
    rds = parse_narccap_filenames(ocgis.env.DIR_DATA)
    ## these are the calculations to perform
    calc = [{'func': 'mean', 'name': 'mean'},
            {'func': 'median', 'name': 'median'},
            {'func': 'max', 'name': 'max'},
            {'func': 'min', 'name': 'min'}]
    ## the temporal grouping to apply when performing the computations
    calc_grouping = ['month', 'year']
    ## the operations object...
    ops = ocgis.OcgOperations(dataset=rds, calc=calc, calc_grouping=calc_grouping,
                              output_format='csv+', geom=geom, abstraction='point',
                              output_crs=CFWGS84)
    print(ops.execute())


if __name__ == '__main__':
    main()
