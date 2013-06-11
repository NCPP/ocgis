from ocgis.test.base import TestBase
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis import env
import os
import itertools
import logging
import ocgis
import webbrowser


class Test(TestBase):

    def test_combinations(self):
        _to_file = [True,False]
        _to_stream = [
#                      True,
                      False
                      ]
        _filename = [os.path.join(env.DIR_OUTPUT,'test_ocgis_log.log'),None]
        _level = [logging.INFO,logging.DEBUG,logging.WARN]
        for to_file,to_stream,filename,level in itertools.product(_to_file,_to_stream,_filename,_level):
            try:
                ocgis_lh.configure(to_file=to_file,filename=filename,to_stream=to_stream)
            except ValueError:
                if to_file and filename is None:
                    continue
                else:
                    raise
            ocgis_lh('a test message')
            subset = ocgis_lh.get_logger('subset')
            interp = ocgis_lh.get_logger('interp')
            ocgis_lh('a subset message',logger=subset)
            ocgis_lh('an interp message',logger=interp)
            ocgis_lh('a general message',alias='foo',ugid=10)
            ocgis_lh('another message',level=level)
            if to_file:
                self.assertTrue(os.path.exists(filename))
                os.remove(filename)
            elif filename is not None:
                self.assertFalse(os.path.exists(filename))
                
    def test_exc(self):
        to_file = True
        to_stream = False
        filename = os.path.join(env.DIR_OUTPUT,'test_ocgis_log.log')
        ocgis_lh.configure(to_file=to_file,to_stream=to_stream,filename=filename)
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
            self.assertTrue(len(lines) > 5)
#        webbrowser.open(log)
