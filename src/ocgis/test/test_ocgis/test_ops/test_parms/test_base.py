from ocgis.exc import DefinitionValidationError
from ocgis.ops.parms.base import AbstractParameter, IterableParameter
from ocgis.test.base import TestBase


class FooAbstractParameter(AbstractParameter):
    name = 'foo_abstract'
    default = 'the_default!'
    nullable = True
    input_types = [str]
    return_type = [str]

    def _get_meta_(self):
        raise NotImplementedError


class TestAbstractParameter(TestBase):
    def test_init(self):
        fp = FooAbstractParameter()
        self.assertEqual(fp.value, FooAbstractParameter.default)

        # test object can be created with an instance of itself
        fp_initial = FooAbstractParameter('the farm')
        fp_second = FooAbstractParameter(fp_initial)
        self.assertEqual(fp_second.value, 'the farm')


class TestIterableParameter(TestBase):
    def get_klass(self):
        class FooIterable(IterableParameter, AbstractParameter):
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
