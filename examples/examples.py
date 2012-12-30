from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.api.operations import OcgOperations
from ocgis.api.interpreter import OcgInterpreter
from ocgis import env


env.WORKSPACE = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/image/agu_for_luca'

sc = ShpCabinet()
gd = sc.get_geom_dict('state_boundaries')

dataset = {'variable':'clt',
           'uri':'http://esg-datanode.jpl.nasa.gov/thredds/dodsC/esg_dataroot/obs4MIPs/observations/atmos/clt/mon/grid/NASA-GSFC/MODIS/v20111130/clt_MODIS_L3_C5_200003-201109.nc'}

ops = OcgOperations(geom=gd,dataset=dataset,snippet=True,output_format='shp',agg_selection=True,vector_wrap=True)
ret1 = OcgInterpreter(ops).execute()
print ret1

ops = OcgOperations(geom=gd,dataset=dataset,snippet=True,output_format='shp',
                    aggregate=True,spatial_operation='clip')
ret2 = OcgInterpreter(ops).execute()
print ret2