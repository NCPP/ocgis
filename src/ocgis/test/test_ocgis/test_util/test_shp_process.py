import os
import shutil
import tempfile

import fiona

from ocgis.spatial.geom_cabinet import GeomCabinet
from ocgis.test.base import TestBase, attr
from ocgis.util.shp_process import ShpProcess


@attr('data')
class TestShpProcess(TestBase):
    def test_shp_process(self):
        copy_path = os.path.join(self.current_dir_output, 'test_shp_process')
        sc = GeomCabinet()
        test_path = os.path.split(sc.get_shp_path('wc_4326'))[0]
        shutil.copytree(test_path, copy_path)

        shp_path = os.path.join(copy_path, 'wc_4326.shp')
        out_folder = tempfile.mkdtemp(dir=self.current_dir_output)
        sp = ShpProcess(shp_path, out_folder)
        sp.process(key='world_countries', ugid=None)

        sc = GeomCabinet(path=out_folder)
        select_ugid = [33, 126, 199]
        geoms = list(sc.iter_geoms('world_countries', select_uid=select_ugid))
        self.assertEqual(len(geoms), 3)
        names = [item['properties']['NAME'] for item in geoms]
        self.assertEqual(set(names), set(['Canada', 'Mexico', 'United States']))

    def test_process_name(self):
        copy_path = os.path.join(self.current_dir_output, 'test_shp_process')
        sc = GeomCabinet()
        test_path = os.path.split(sc.get_shp_path('wc_4326'))[0]
        shutil.copytree(test_path, copy_path)

        shp_path = os.path.join(copy_path, 'wc_4326.shp')
        out_folder = tempfile.mkdtemp(dir=self.current_dir_output)
        sp = ShpProcess(shp_path, out_folder)
        sp.process(key='world_countries', ugid=None, name='new_id')
        path = os.path.join(out_folder, 'world_countries.shp')
        with fiona.open(path, 'r') as sci:
            uids = [record['properties']['new_id'] for record in sci]
        self.assertEqual(uids, range(1, 212))
