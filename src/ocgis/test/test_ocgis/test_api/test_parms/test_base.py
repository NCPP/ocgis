from ocgis.api.parms.base import OcgParameter, IterableParameter
from ocgis.exc import DefinitionValidationError
from ocgis.test.base import TestBase


class TestIterableParameter(TestBase):

    def get_klass(self):

        class FooIterable(IterableParameter, OcgParameter):
            name = 'foo_iterable'
            element_type = str
            unique = True
            default = None
            nullable = False
            input_types = [list, tuple]
            return_type = tuple

            def _get_meta_(self):
                pass

        return FooIterable

    def test_constructor(self):
        klass = self.get_klass()
        ff = klass(['hi_there'])
        self.assertEqual(ff.value, ('hi_there',))
        self.assertEqual(klass(['tas', 'tasmax']).value, ('tas', 'tasmax'))
        self.assertEqual(klass(('tas', 'tasmax')).value, ('tas', 'tasmax'))

    def test_unique(self):
        klass = self.get_klass()
        with self.assertRaises(DefinitionValidationError):
            klass(['hi', 'hi'])
