from ocgis.api.interp.iocg.interpreter_ocg import OcgInterpreter
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.util.inspect import Inspect


########## GLOBALS ##################################
## path to nc file
NC = 'TMAXdaysAboveThreshold1950_1955ByGridpoint.nc'
## variable to extract from the nc
VARIABLE = 'gmo_tmax-days_above_threshold'
## directory containing shapefiles
SHP = 'shp'
## name of the folder containing the target shapefile
KEY = 'NC CSC region'
#####################################################

def main():
    ## inspect the dataset
#    inspect = Inspect(NC)

    ## object to manage shapefile geometries
    sc = ShpCabinet(path=SHP)
    ## extract geometry dictionary from shapefile
    geom_dict = sc.get_geom_dict(KEY)
    ## update id field (required by OCGIS)
    for geom in geom_dict:
        geom['id'] = geom['CSC_ID']
    ## operations dictionary
    ops = {'meta':[{'uri':NC,'variable':VARIABLE}],
           'time_range':None,
           'level_range':None,
           'spatial_operation':'intersects',
           'output_format':'shp',
           'geom':geom_dict,
           'aggregate':False}
    ## execute the operations returning a path to the directory
    ## containing the output
    ret = OcgInterpreter(ops).execute()

    return(ret)


if __name__ == '__main__':
    ret = main()
    print(ret)
