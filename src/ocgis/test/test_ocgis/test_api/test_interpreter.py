import logging
import os

from ocgis import OcgOperations
from ocgis import constants, env
from ocgis.api.interpreter import OcgInterpreter
from ocgis.api.subset import OperationsEngine
from ocgis.conv.fiona_ import ShpConverter
from ocgis.conv.nc import NcConverter
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.exc import ExtentError
from ocgis.test.base import TestBase
from ocgis.test.base import attr
from ocgis.util.itester import itr_products_keywords
from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations


class TestOcgInterpreter(TestBase):
    @attr('data')
    def test_execute_directory(self):
        """Test that the output directory is removed appropriately following an operations failure."""

        kwds = dict(add_auxiliary_files=[True, False])
        rd = self.test_data.get_rd('cancm4_tas')

        # this geometry is outside the domain and will result in an exception
        geom = [1000, 1000, 1100, 1100]

        for k in itr_products_keywords(kwds, as_namedtuple=True):
            ops = OcgOperations(dataset=rd, output_format='csv', add_auxiliary_files=k.add_auxiliary_files, geom=geom)
            try:
                ops.execute()
            except ExtentError:
                contents = os.listdir(self.current_dir_output)
                self.assertEqual(len(contents), 0)

    @attr('data')
    def test_get_converter(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        outdir = self.current_dir_output
        prefix = 'foo'
        interp = OcgInterpreter(ops)
        so = OperationsEngine(ops)
        ret = interp._get_converter_(NumpyConverter, outdir, prefix, so)
        self.assertIsInstance(ret, NumpyConverter)

        # Test melted is registered by the converter.
        ops = OcgOperations(dataset=rd, melted=True, output_format=constants.OUTPUT_FORMAT_SHAPEFILE)
        interp = OcgInterpreter(ops)
        ret = interp._get_converter_(ShpConverter, outdir, prefix, so)
        self.assertIsInstance(ret, ShpConverter)
        self.assertTrue(ret.melted)

        # Test options are passed to the underlying converter.
        opts = {'data_model': 'foo'}
        ops = OcgOperations(dataset=rd, output_format='nc', output_format_options=opts)
        interp = OcgInterpreter(ops)
        ret = interp._get_converter_(NcConverter, outdir, prefix, so)
        self.assertDictEqual(ret.options, opts)

    @attr('data')
    def test_get_progress_and_configure_logging(self):
        env.VERBOSE = True
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        outdir = self.current_dir_output
        prefix = 'foo'
        interp = OcgInterpreter(ops)
        self.assertIsNone(logging._warnings_showwarning)
        self.assertTrue(ocgis_lh.null)
        env.SUPPRESS_WARNINGS = False
        progress = interp._get_progress_and_configure_logging_(outdir, prefix)
        self.assertIsInstance(progress, ProgressOcgOperations)
        self.assertFalse(ocgis_lh.null)
        self.assertFalse(logging._warnings_showwarning)
