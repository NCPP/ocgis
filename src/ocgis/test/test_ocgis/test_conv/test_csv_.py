import os
import tempfile
from csv import DictReader

import fiona
import numpy as np

from ocgis import OcgOperations, RequestDataset
from ocgis import constants, env
from ocgis.conv.csv_ import CsvShapefileConverter, CsvConverter
from ocgis.ops.engine import OperationsEngine
from ocgis.test.base import attr
from ocgis.test.test_ocgis.test_conv.test_base import AbstractTestConverter
from ocgis.util.addict import Dict


class TestCsvConverter(AbstractTestConverter):
    def get(self, kwargs_conv=None, kwargs_ops=None):
        rd = self.test_data.get_rd('cancm4_tas')

        kwds_ops = Dict(dataset=rd, geom='state_boundaries', select_ugid=[15, 18], snippet=True)
        if kwargs_ops is not None:
            kwds_ops.update(kwargs_ops)

        ops = OcgOperations(**kwds_ops)
        so = OperationsEngine(ops)

        kwds_conv = Dict()
        kwds_conv.outdir = self.current_dir_output
        kwds_conv.prefix = 'foo'
        kwds_conv.ops = ops
        if kwargs_conv is not None:
            kwds_conv.update(kwargs_conv)

        conv = CsvConverter(so, **kwds_conv)

        return conv

    @attr('data')
    def test_write(self):
        conv = self.get()
        self.assertFalse(conv.melted)
        ret = conv.write()
        ugids = []
        with open(ret) as f:
            reader = DictReader(f)
            for row in reader:
                ugids.append(row[constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER])
        self.assertAsSetEqual(['15', '18'], ugids)


class TestCsvShapefileConverter(AbstractTestConverter):
    def get(self, kwargs_conv=None, kwargs_ops=None):
        rd = self.test_data.get_rd('cancm4_tas')

        kwds_ops = Dict(dataset=rd, geom='state_boundaries', select_ugid=[15, 18], snippet=True)
        if kwargs_ops is not None:
            kwds_ops.update(kwargs_ops)

        ops = OcgOperations(**kwds_ops)
        so = OperationsEngine(ops)

        kwds_conv = Dict()
        kwds_conv.outdir = self.current_dir_output
        kwds_conv.prefix = 'foo'
        kwds_conv.ops = ops
        if kwargs_conv is not None:
            kwds_conv.update(kwargs_conv)

        conv = CsvShapefileConverter(so, **kwds_conv)

        return conv

    @attr('data')
    def test_init(self):
        conv = self.get()
        self.assertIsInstance(conv, CsvConverter)

    @attr('data')
    def test_build(self):
        path = self.get_shapefile_path_with_no_ugid()
        keywords = dict(geom_uid=['ID', None])
        rd = self.test_data.get_rd('cancm4_tas')
        for k in self.iter_product_keywords(keywords):
            if k.geom_uid is None:
                geom_select_uid = None
            else:
                geom_select_uid = [8]
            ops = OcgOperations(dataset=rd, geom=path, geom_uid=k.geom_uid, geom_select_uid=geom_select_uid,
                                snippet=True)
            coll = ops.execute()
            conv = CsvShapefileConverter([coll], outdir=self.current_dir_output, prefix='foo', overwrite=True, ops=ops)
            ret = conv._build_(coll)

            if k.geom_uid is None:
                actual = env.DEFAULT_GEOM_UID
            else:
                actual = k.geom_uid
            actual = [constants.HEADERS.ID_DATASET.upper(), actual, constants.HEADERS.ID_GEOMETRY.upper()]
            self.assertEqual(actual, ret['fiona_object'].meta['schema']['properties'].keys())

    @attr('data')
    def test_geom_uid(self):
        rd = self.test_data.get_rd('cancm4_tas')
        for geom_uid in ['IDD', None]:
            ops = OcgOperations(dataset=rd, geom_uid=geom_uid)
            conv = CsvShapefileConverter(None, ops=ops, outdir=self.current_dir_output, prefix='foo')
            if geom_uid is None:
                geom_uid = env.DEFAULT_GEOM_UID
            self.assertEqual(conv.geom_uid, geom_uid)

    @attr('data')
    def test_write(self):
        # test melted format
        for melted in [False, True]:
            kwargs_ops = dict(melted=melted)
            kwargs_conv = dict(outdir=tempfile.mkdtemp(dir=self.current_dir_output))

            conv = self.get(kwargs_ops=kwargs_ops, kwargs_conv=kwargs_conv)
            csv_path = conv.write()
            self.assertTrue(os.path.exists(csv_path))
            self.assertEqual(conv._ugid_gid_store,
                             {1: {18: [5988, 5989, 5990, 6116, 6117, 6118], 15: [5992, 6119, 6120]}})

            shp_path = os.path.split(csv_path)[0]
            shp_path = os.path.join(shp_path, 'shp')
            shp_path_gid = os.path.join(shp_path, 'foo_gid.shp')
            target = RequestDataset(shp_path_gid).get()
            self.assertEqual(target.shape[-1], 9)
            shp_path_ugid = os.path.join(shp_path, 'foo_ugid.shp')
            target = RequestDataset(shp_path_ugid).get()
            self.assertEqual(target.shape[-1], 2)

        # test aggregating the selection geometry
        rd1 = self.test_data.get_rd('cancm4_tasmax_2011')
        rd2 = self.test_data.get_rd('maurer_bccr_1950')

        keywords = dict(agg_selection=[True, False])
        for k in self.iter_product_keywords(keywords):
            ops = OcgOperations(dataset=[rd1, rd2], snippet=True, output_format='csv-shp', geom='state_boundaries',
                                agg_selection=k.agg_selection, select_ugid=[32, 47], prefix=str(k.agg_selection))
            ret = ops.execute()
            directory = os.path.split(ret)[0]

            path_ugid = os.path.join(directory, 'shp', '{0}_ugid.shp'.format(ops.prefix))
            with fiona.open(path_ugid) as source:
                records = list(source)
            if k.agg_selection:
                uids = [1]
            else:
                uids = [32, 47]
            self.assertEqual([r['properties'][env.DEFAULT_GEOM_UID] for r in records], uids)

            path_gid = os.path.join(directory, 'shp', '{0}_gid.shp'.format(ops.prefix))
            with fiona.open(path_gid) as source:
                uid = [r['properties'][env.DEFAULT_GEOM_UID] for r in source]
            if k.agg_selection:
                self.assertAsSetEqual(uid, [1])
            else:
                uid = np.array(uid)
                self.assertEqual(np.sum(uid == 32), 1915)
                self.assertEqual(np.sum(uid == 47), 923)

            meta = os.path.join(os.path.split(ret)[0], '{0}_source_metadata.txt'.format(ops.prefix))
            with open(meta, 'r') as f:
                lines = f.readlines()
            self.assertTrue(len(lines) > 50)
