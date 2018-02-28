import logging
import os
import shutil
from ConfigParser import SafeConfigParser
from tempfile import mkdtemp

from ocgis.util.shp_process import ShpProcess

shape_folders = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/shp'
logging.basicConfig(level=logging.INFO)
log_fiona = logging.getLogger('Fiona')
log_fiona.setLevel(logging.ERROR)

output_folder = mkdtemp(prefix='shp')
logging.info('output folder = ' + output_folder)
for dirpath, dirnames, filenames in os.walk(shape_folders):
    for filename in filenames:
        if filename.endswith('.shp'):
            logging.info(filename)
            shp_path = os.path.join(dirpath, filename)
            cfg_path = shp_path.replace('.shp', '.cfg')
            key = filename.split('.')[0]
            config = SafeConfigParser()
            config.read(cfg_path)
            ugid = config.get('mapping', 'ugid')
            if ugid == 'none':
                ugid = None
            sp = ShpProcess(shp_path)
            try:
                sp.process(output_folder, key, ugid=ugid)
            except KeyError:
                sp.process(output_folder, key, ugid=ugid.upper())
            new_cfg_path = os.path.join(output_folder, key, os.path.split(cfg_path)[1])
            shutil.copy2(cfg_path, new_cfg_path)

logging.info('success')
