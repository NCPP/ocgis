from collections import OrderedDict

from ocgis.ops.parms import MetadataAttributes
from ocgis.test.base import TestBase


class TestMetadataAttributes(TestBase):
    def test_init(self):
        ma = MetadataAttributes()
        for k in ma._keys:
            self.assertEqual(ma.value[k], {})

        ma = MetadataAttributes({'first': 56})
        self.assertDictEqual(ma.value, {'variable': {'first': 56}, 'field': {}})

    def test_merge(self):
        ms = MetadataAttributes()
        other = OrderedDict({'variable': 'name', 'field': 5})
        ms.merge(other)
        actual = {'variable': {'variable': 'name', 'field': 5}, 'field': {}}
        self.assertDictEqual(ms.value, actual)
        self.assertIsInstance(ms.value['variable'], OrderedDict)
        other['variable'] = 'foo'
        self.assertDictEqual(ms.value, actual)
