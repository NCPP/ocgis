from copy import deepcopy
from unittest import SkipTest

import ESMF
import numpy as np

from ocgis.exc import DefinitionValidationError
from ocgis import SpatialCollection, OcgOperations
from ocgis.conv.base import AbstractConverter
from ocgis.conv.esmpy import ESMPyConverter
from ocgis.test.test_ocgis.test_conv.test_base import AbstractTestConverter


class TestESMPyConverter(AbstractTestConverter):

    def setUp(self):
        raise SkipTest

    def get_conv(self, with_corners=True, value_mask=None, esmf_field_name=None, field=None):
        coll = self.get_spatial_collection(field=field)
        conv = ESMPyConverter([coll], with_corners=with_corners, value_mask=value_mask, esmf_field_name=esmf_field_name)
        return conv

    def test_init(self):
        conv = self.get_conv()
        self.assertIsInstance(conv, AbstractConverter)

    def test_iter(self):
        conv = self.get_conv()
        res = list(conv)
        self.assertEqual(len(res), 1)
        self.assertIsInstance(res[0], SpatialCollection)

    def test_write(self):
        #todo: test with multiple collections
        #todo: test with multiple variables
        #todo: test with multiple fields
        #todo: test with mask on field

        kwds = dict(nlevel=[1, 3],
                    nrlz=[1, 5],
                    esmf_field_name=[None, 'foo'],
                    value_mask=[None, True])

        for k in self.iter_product_keywords(kwds):
            ofield = self.get_field(nlevel=k.nlevel, nrlz=k.nrlz)
            ovariable = ofield.variables.first()

            if k.value_mask is not None:
                value_mask = np.zeros(ofield.shape[-2:], dtype=bool)
                value_mask[0, 1] = True
            else:
                value_mask = None

            conv = self.get_conv(field=ofield, with_corners=True, value_mask=value_mask,
                                 esmf_field_name=k.esmf_field_name)
            efield = conv.write()

            try:
                self.assertEqual(efield.name, k.esmf_field_name)
            except AssertionError:
                self.assertIsNone(k.esmf_field_name)
                self.assertEqual(efield.name, ovariable.alias)
            self.assertIsInstance(efield, ESMF.Field)
            self.assertFalse(np.may_share_memory(ovariable.value, efield))
            # field are currently always 64-bit in ESMF...
            self.assertEqual(efield.dtype, np.float64)
            self.assertNumpyAll(ovariable.value, efield, check_arr_type=False, check_arr_dtype=False)

            if k.value_mask:
                self.assertTrue(np.any(efield.grid.mask[0] == 0))
                self.assertTrue(efield.mask.any())

    def test_validate_ops(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = deepcopy(rd)
        rd2.alias = 'tas2'
        dataset = [rd, rd2]

        # only one dataset for output
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=dataset, output_format='esmpy')

        # clip not allowed
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, output_format='esmpy', spatial_operation='clip')

        # only one select_ugid
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, output_format='esmpy', geom='state_boundaries', select_ugid=[4, 5])
        # more than one is allowed if agg_selection is true
        OcgOperations(dataset=rd, output_format='esmpy', geom='state_boundaries', select_ugid=[4, 5],
                      agg_selection=True)

        # no spatial aggregation
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd, output_format='esmpy', aggregate=True)