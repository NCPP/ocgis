import ocgis

ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True
ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/conference/FOSS4G_2013/figures/rcm-gcm'

cancm4 = ocgis.RequestDataset(
    uri='/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/CanCM4/tas_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
    variable='tas')
narccap = ocgis.RequestDataset(uri='/home/local/WX/ben.koziol/climate_data/narccap/tas_CRCM_cgcm3_1981010103.nc',
                               variable='tas')
import ipdb;

ipdb.set_trace()
todo = [[cancm4, 'cancm4'],
        [narccap, 'narccap']]
for td in todo:
    ops = ocgis.OcgOperations(dataset=td[0], prefix=td[1], snippet=True, output_format='shp')
    ops.execute()
