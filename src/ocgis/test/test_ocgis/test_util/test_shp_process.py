from ocgis.test.base import TestBase
import os
import shutil
from ocgis.util.shp_process import ShpProcess
import tempfile
from ocgis.util.shp_cabinet import ShpCabinet


class Test(TestBase):
    
    def test_shp_process(self):
        test_path = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/test_data/test_shp_process'
        copy_path = os.path.join(self._test_dir,'test_shp_process')
        shutil.copytree(test_path,copy_path)
        shp_path = os.path.join(copy_path,'wc_4326.shp')
        out_folder = tempfile.mkdtemp(dir=self._test_dir)
        sp = ShpProcess(shp_path,out_folder)
        sp.process(key='world_countries',ugid=None)
            
        sc = ShpCabinet(path=out_folder)
        select_ugid = [33,126,199]
        geoms = list(sc.iter_geoms('world_countries',select_ugid=select_ugid))
        self.assertEqual(len(geoms),3)
        names = [item['properties']['NAME'] for item in geoms]
        self.assertEqual(set(names),set(['Canada','Mexico','United States']))
