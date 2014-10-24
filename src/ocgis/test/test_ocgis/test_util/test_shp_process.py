from ocgis.test.base import TestBase
import os
import shutil
from ocgis.util.shp_process import ShpProcess
import tempfile
from ocgis.util.shp_cabinet import ShpCabinet
import subprocess


class TestShpProcess(TestBase):

    def test_shp_process(self):
        copy_path = os.path.join(self.current_dir_output, 'test_shp_process')
        sc = ShpCabinet()
        test_path = os.path.split(sc.get_shp_path('wc_4326'))[0]
        shutil.copytree(test_path, copy_path)

        shp_path = os.path.join(copy_path, 'wc_4326.shp')
        out_folder = tempfile.mkdtemp(dir=self.current_dir_output)
        sp = ShpProcess(shp_path, out_folder)
        sp.process(key='world_countries', ugid=None)

        sc = ShpCabinet(path=out_folder)
        select_ugid = [33, 126, 199]
        geoms = list(sc.iter_geoms('world_countries', select_ugid=select_ugid))
        self.assertEqual(len(geoms), 3)
        names = [item['properties']['NAME'] for item in geoms]
        self.assertEqual(set(names), set(['Canada', 'Mexico', 'United States']))
