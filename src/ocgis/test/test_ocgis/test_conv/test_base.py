import itertools
import os
import tempfile
from copy import deepcopy
from csv import DictReader

import numpy as np

import ocgis
from ocgis import SpatialCollection
from ocgis import constants
from ocgis.constants import OutputFormatName
from ocgis.conv.base import AbstractTabularConverter, get_converter_map, AbstractCollectionConverter, \
    AbstractFileConverter
from ocgis.conv.csv_ import CsvConverter, CsvShapefileConverter
from ocgis.conv.fiona_ import ShpConverter, GeoJsonConverter
from ocgis.conv.meta import MetaJSONConverter
from ocgis.conv.nc import NcConverter
from ocgis.test.base import TestBase, nc_scope
from ocgis.test.base import attr


class AbstractTestConverter(TestBase):
    def get_spatial_collection(self, field=None):
        rd = self.test_data.get_rd('cancm4_tas')
        # field = field or rd.get()[:, 0, :, 0, 0]
        if field is None:
            field = rd.get().get_field_slice({'time': 0, 'x': 0, 'y': 0})
        coll = SpatialCollection()
        coll.add_field(field, None)
        return coll


class Test(TestBase):
    def test_get_converter_map(self):
        cmap = get_converter_map()
        self.assertEqual(cmap[constants.OutputFormatName.METADATA_JSON], MetaJSONConverter)


class FakeAbstractCollectionConverter(AbstractCollectionConverter):
    _add_ugeom = True

    def _build_(self, *args, **kwargs):
        pass

    def _finalize_(self, *args, **kwargs):
        pass

    def _write_coll_(self, f, coll):
        pass


class TestAbstractCollectionConverter(AbstractTestConverter):
    _auxiliary_file_list = ['ocgis_output_metadata.txt', 'ocgis_output_source_metadata.txt', 'ocgis_output_did.csv']

    def run_auxiliary_file_tst(self, Converter, file_list, auxiliary_file_list=None):
        auxiliary_file_list = auxiliary_file_list or self._auxiliary_file_list
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, output_format=OutputFormatName.OCGIS,
                                  slice=[None, 0, None, [0, 10], [0, 10]])
        coll = ops.execute()

        _ops = [None, ops]
        _add_auxiliary_files = [True, False]
        for ops_arg, add_auxiliary_files in itertools.product(_ops, _add_auxiliary_files):
            # # make a new output directory as to not deal with overwrites
            outdir = tempfile.mkdtemp(dir=self.current_dir_output)
            try:
                conv = Converter([coll], outdir=outdir, prefix='ocgis_output', add_auxiliary_files=add_auxiliary_files,
                                 ops=ops_arg)
            # # CsvShapefileConverter requires an operations argument
            except ValueError as e:
                if Converter == CsvShapefileConverter and ops_arg is None:
                    continue
                else:
                    raise e
            conv.write()
            files = os.listdir(outdir)
            # # auxiliary files require an operations argument
            if add_auxiliary_files == True and ops_arg is not None:
                to_test = deepcopy(file_list)
                to_test.extend(auxiliary_file_list)
            else:
                to_test = file_list
            self.assertEqual(set(files), set(to_test))

    def run_overwrite_true_tst(self, Converter, include_ops=False):
        rd = self.test_data.get_rd('cancm4_tas')
        _ops = ocgis.OcgOperations(dataset=rd, output_format=OutputFormatName.OCGIS,
                                   slice=[None, 0, None, [0, 10], [0, 10]])
        coll = _ops.execute()

        ops = _ops if include_ops else None
        outdir = tempfile.mkdtemp(dir=self.current_dir_output)
        conv = Converter([coll], outdir=outdir, prefix='ocgis_output', ops=ops)
        conv.write()
        mtimes = [os.path.getmtime(os.path.join(outdir, f)) for f in os.listdir(outdir)]

        Converter([coll], outdir=outdir, prefix='ocgis_output', overwrite=True, ops=ops).write()
        mtimes2 = [os.path.getmtime(os.path.join(outdir, f)) for f in os.listdir(outdir)]
        # # if the file is overwritten the modification time will be more recent!
        self.assertTrue(all([m2 > m for m2, m in zip(mtimes2, mtimes)]))

    @attr('data')
    def test_multiple_variables(self):
        conv_klasses = [CsvConverter, NcConverter]
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        var2 = deepcopy(field['tas'].extract())
        var2.set_name('tas2')
        field.add_variable(var2, is_data=True)
        field = field.get_field_slice({'time': slice(0, 2), 'y': slice(0, 5), 'x': slice(0, 5)})
        coll = self.get_spatial_collection(field=field)
        for conv_klass in conv_klasses:
            if conv_klass == CsvConverter:
                kwds = {'melted': True}
            else:
                kwds = {}
            prefix = 'ocgis_output_{0}'.format(conv_klass.__name__)
            conv = conv_klass([coll], outdir=self.current_dir_output, prefix=prefix, **kwds)
            ret = conv.write()
            if conv_klass == CsvConverter:
                with open(ret, 'r') as f:
                    reader = DictReader(f)
                    aliases = set([row['VARIABLE'] for row in reader])
                    self.assertEqual(set(['tas', 'tas2']), aliases)
            else:
                with nc_scope(ret) as ds:
                    self.assertAlmostEqual(ds.variables['tas'][:].mean(), np.float32(247.08411))
                    self.assertNumpyAll(ds.variables['tas'][:], ds.variables['tas2'][:])

    @attr('data')
    def test_overwrite_false_csv(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, output_format=OutputFormatName.OCGIS,
                                  slice=[None, 0, None, [0, 10], [0, 10]])
        coll = ops.execute()

        outdir = tempfile.mkdtemp(dir=self.current_dir_output)
        conv = CsvConverter([coll], outdir=outdir, prefix='ocgis_output')
        conv.write()

        with self.assertRaises(IOError):
            CsvConverter([coll], outdir=outdir, prefix='ocgis_output')

    @attr('data')
    def test_overwrite_true_csv(self):
        self.run_overwrite_true_tst(CsvConverter)

    @attr('data')
    def test_overwrite_true_nc(self):
        self.run_overwrite_true_tst(NcConverter)

    @attr('data')
    def test_overwrite_true_shp(self):
        self.run_overwrite_true_tst(ShpConverter)

    @attr('data')
    def test_overwrite_true_csv_shp(self):
        self.run_overwrite_true_tst(CsvShapefileConverter, include_ops=True)

    @attr('data')
    def test_add_auxiliary_files_csv(self):
        self.run_auxiliary_file_tst(CsvConverter, ['ocgis_output.csv'])

    @attr('data')
    def test_add_auxiliary_files_geojson(self):
        self.run_auxiliary_file_tst(GeoJsonConverter, ['ocgis_output.json'])

    @attr('data')
    def test_add_auxiliary_files_nc(self):
        self.run_auxiliary_file_tst(NcConverter, ['ocgis_output.nc'])

    @attr('data')
    def test_add_auxiliary_files_csv_shp(self):
        self.run_auxiliary_file_tst(CsvShapefileConverter, ['ocgis_output.csv', 'shp'])

    @attr('data')
    def test_add_auxiliary_files_shp(self):
        self.run_auxiliary_file_tst(ShpConverter,
                                    ['ocgis_output.dbf', 'ocgis_output.shx', 'ocgis_output.shp', 'ocgis_output.cpg',
                                     'ocgis_output.prj'])


class FakeAbstractTabularConverter(AbstractTabularConverter):
    pass


class TestAbstractTabularConverter(AbstractTestConverter):
    def test_init(self):
        ff = FakeAbstractTabularConverter(None)
        self.assertIsInstance(ff, AbstractFileConverter)
        self.assertFalse(ff.melted)

        ff = FakeAbstractTabularConverter(None, melted=True)
        self.assertTrue(ff.melted)
