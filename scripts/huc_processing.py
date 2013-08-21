import logging
import sys
import os


'''
Process HUC data from geodatabase format to a format acceptable for OCGIS.

Original zipped HUC data was downloaded in zipped GDB format from: 
 ftp://ftp.ftw.nrcs.usda.gov/wbd/WBD_Latest_Version_June2013/
'''

DIR_GDBDATA = r'F:\htmp\huc'
DIR_OUTPUT = r'F:\htmp\huc\as_shp'
DIR_OUTPUT_LINUX = '/home/local/WX/ben.koziol/htmp/huc/as_shp'
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('gdb.convert')
GDBS = [
#        'WBDHU10_June2013.gdb',
#        'WBDHU4_June2013.gdb',
#        'WBDHU12_June2013.gdb',
#        'WBDHU8_June2013.gdb',
#        'WBD_Line_June2013.gdb',
        'WBDHU2_June2013.gdb',
#        'WBDHU6_June2013.gdb'
        ]

def convert_to_shapefile():
    LOG.info('starting...')
    
    import arcpy
    from arcpy import env
    
    env.workspace = DIR_GDBDATA
        
    for gdb in GDBS:
        fc = gdb.split('_')[0]
        LOG.info('converting --> '+gdb+'/'+fc)
        out_shp_name = os.path.splitext(gdb)[0] + '.shp'
        LOG.info(' writing to --> '+os.path.join(DIR_OUTPUT,out_shp_name))
        arcpy.FeatureClassToFeatureClass_conversion(os.path.join(gdb,fc), 
                                                    DIR_OUTPUT, 
                                                    out_shp_name)
        LOG.info(' complete.')
    LOG.info('success.')
    
def prepare_shapefile(filename):
    from ocgis.util.shp_process import ShpProcess
    
    LOG.info('preparing shapefile --> '+filename)
    shp_path = os.path.join(DIR_OUTPUT_LINUX,filename)
    sp = ShpProcess(shp_path)
    key = os.path.splitext(filename)[0]
    sp.process(DIR_OUTPUT_LINUX,key)
    LOG.info('success.')


if __name__ == '__main__':
#    convert_to_shapefile()
    prepare_shapefile('WBDHU2_June2013.shp')