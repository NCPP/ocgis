from ocgis.test.base import TestBase
from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations
from ocgis import env
import os
import itertools
import logging
import ocgis
from ocgis.util.helpers import get_temp_path


class TestProgressOcgOperations(TestBase):
    
    def test_constructor(self):
        prog = ProgressOcgOperations(lambda x,y: (x,y))
        self.assertEqual(prog.n_operations,1)
        self.assertEqual(prog.n_completed_operations,0)
    
    def test_simple(self):
        n_geometries = 3
        n_calculations = 3
        
        def callback(percent,message):
            return(percent,message)
        
        for cb in [callback,None]:
            prog = ProgressOcgOperations(callback=cb,n_geometries=n_geometries,n_calculations=n_calculations)
            n_operations = 9
            self.assertEqual(prog.n_operations,n_operations)
            prog.mark()
            if cb is None:
                self.assertEqual(prog.percent_complete,100*(1/float(n_operations)))
                self.assertEqual(prog(),None)
            else:
                self.assertEqual(prog(),(100*(1/float(n_operations)),None))
            prog.mark()
            if cb is None:
                self.assertEqual(prog.percent_complete,(100*(2/float(n_operations))))
            else:
                self.assertEqual(prog(message='hi'),(100*(2/float(n_operations)),'hi'))
        
    def test_hypothetical_operations_loop(self):
        
        def callback(percent,message):
            return(percent,message)
        
        n = [0,1,2]
        for n_subsettables,n_geometries,n_calculations in itertools.product(n,n,n):
            try:
                prog = ProgressOcgOperations(callback,
                                             n_subsettables=n_subsettables,
                                             n_geometries=n_geometries,
                                             n_calculations=n_calculations)
            except AssertionError:
                if n_geometries == 0 or n_subsettables == 0:
                    continue
                else:
                    raise
            
            for ns in range(n_subsettables):
                for ng in range(n_geometries):
                    for nc in range(n_calculations):
                        prog.mark()
                    if n_calculations == 0:
                        prog.mark()
            self.assertEqual(prog(),(100.0,None))


class TestOcgisLogging(TestBase):
    
    def tearDown(self):
        ocgis_lh.shutdown()
        TestBase.tearDown(self)
    
    def test_with_callback(self):
        fp = get_temp_path(wd=self._test_dir)
                
        def callback(message,path=fp):
            with open(fp,'a') as sink:
                sink.write(message)
                sink.write('\n')
                
        class FooError(Exception):
            pass
            
        ocgis_lh.configure(callback=callback)
        ocgis_lh(msg='this is a test message')
        ocgis_lh()
        ocgis_lh(msg='this is a second test message')
        ocgis_lh(msg='this should not be there',level=logging.DEBUG)
        exc = FooError('foo message for value error')
        try:
            ocgis_lh(exc=exc)
        except FooError:
            pass
        with open(fp,'r') as source:
            lines = source.readlines()
        self.assertEqual(lines,['this is a test message\n', 'this is a second test message\n', 'FooError: foo message for value error\n'])
    
    def test_simple(self):
        to_file = os.path.join(env.DIR_OUTPUT,'test_ocgis_log.log')
        to_stream = False
        
        ocgis_lh.configure(to_file,to_stream)
        
        ocgis_lh('a test message')
        subset = ocgis_lh.get_logger('subset')
        subset.info('a subset message')
        

    def test_combinations(self):
        _to_stream = [
#                      True,
                      False
                      ]
        _to_file = [
                    os.path.join(env.DIR_OUTPUT,'test_ocgis_log.log'),
                    None
                    ]
        _level = [logging.INFO,logging.DEBUG,logging.WARN]
        for ii,(to_file,to_stream,level) in enumerate(itertools.product(_to_file,_to_stream,_level)):
            ocgis_lh.configure(to_file=to_file,to_stream=to_stream,level=level)
            try:
                ocgis_lh(ii)
                ocgis_lh('a test message')
                subset = ocgis_lh.get_logger('subset')
                interp = ocgis_lh.get_logger('interp')
                ocgis_lh('a subset message',logger=subset)
                ocgis_lh('an interp message',logger=interp)
                ocgis_lh('a general message',alias='foo',ugid=10)
                ocgis_lh('another message',level=level)
                if to_file is not None:
                    self.assertTrue(os.path.exists(to_file))
                    os.remove(to_file)
            finally:
                logging.shutdown()
                
    def test_exc(self):
        to_file = os.path.join(env.DIR_OUTPUT,'test_ocgis_log.log')
        to_stream = False
        ocgis_lh.configure(to_file=to_file,to_stream=to_stream)
        try:
            raise(ValueError('some exception information'))
        except Exception as e:
            with self.assertRaises(ValueError):
                ocgis_lh('something happened',exc=e)
                
    def test_writing(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='csv')
        ret = ops.execute()
        folder = os.path.split(ret)[0]
        log = os.path.join(folder,ops.prefix+'.log')
        with open(log) as f:
            lines = f.readlines()
            self.assertTrue(len(lines) >= 4)
