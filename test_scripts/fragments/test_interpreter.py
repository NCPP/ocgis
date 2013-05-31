import unittest
from ocg.api.interp.interpreter import Interpreter
from ocg.test.misc import gen_descriptor_classes, pause_test


class TestInterpreter(unittest.TestCase):
    
    @pause_test 
    def test_check(self):
        for desc in gen_descriptor_classes():
            interp = Interpreter.get_interpreter(desc)
            interp.check()
            
    def test_execute(self):
        for desc in gen_descriptor_classes(niter=1):
            interp = Interpreter.get_interpreter(desc)
            interp.execute()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()