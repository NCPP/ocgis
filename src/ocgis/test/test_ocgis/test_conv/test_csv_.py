from csv import DictReader
import os
import tempfile

from ocgis import constants

from ocgis.conv.csv_ import CsvShapefileConverter, CsvConverter
from ocgis import OcgOperations, RequestDataset
from ocgis.api.subset import SubsetOperation
from ocgis.test.test_ocgis.test_conv.test_base import AbstractTestConverter
from ocgis.util.addict import Dict


class TestCsvConverter(AbstractTestConverter):
    def get(self, kwargs_conv=None, kwargs_ops=None):
        rd = self.test_data.get_rd('cancm4_tas')

        kwds_ops = Dict(dataset=rd, geom='state_boundaries', select_ugid=[15, 18], snippet=True)
        if kwargs_ops is not None:
            kwds_ops.update(kwargs_ops)

        ops = OcgOperations(**kwds_ops)
        so = SubsetOperation(ops)

        kwds_conv = Dict()
        kwds_conv.outdir = self.current_dir_output
        kwds_conv.prefix = 'foo'
        kwds_conv.ops = ops
        if kwargs_conv is not None:
            kwds_conv.update(kwargs_conv)

        conv = CsvConverter(so, **kwds_conv)

        return conv

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


class TestCsvShpConverter(AbstractTestConverter):
    def get(self, kwargs_conv=None, kwargs_ops=None):
        rd = self.test_data.get_rd('cancm4_tas')

        kwds_ops = Dict(dataset=rd, geom='state_boundaries', select_ugid=[15, 18], snippet=True)
        if kwargs_ops is not None:
            kwds_ops.update(kwargs_ops)

        ops = OcgOperations(**kwds_ops)
        so = SubsetOperation(ops)

        kwds_conv = Dict()
        kwds_conv.outdir = self.current_dir_output
        kwds_conv.prefix = 'foo'
        kwds_conv.ops = ops
        if kwargs_conv is not None:
            kwds_conv.update(kwargs_conv)

        conv = CsvShapefileConverter(so, **kwds_conv)

        return conv

    def test_init(self):
        conv = self.get()
        self.assertIsInstance(conv, CsvConverter)

    def test(self):
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