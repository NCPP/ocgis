import csv
from ocgis.conv.converter import OcgConverter
from ocgis.conv.csv_ import OcgDialect
from ocgis.util.helpers import get_temp_path
from ocgis.conv.shpidx import ShpIdxIdentifierConverter
import os.path
from ocgis.api.dataset.collection.iterators import KeyedIterator


class KeyedConverter(OcgConverter):
    _ext = 'csv'
    
#    def __init__(self,*args,**kwds):
#        super(KeyedConverter,self).__init__(*args,**kwds)
#        self.wd = get_temp_path(only_dir=True,wd=self.wd,nest=True)
    
    def _write_iter_dict_(self,dct):
        for k,v in dct.iteritems():
            with open(self._get_path_(k),'w') as f:
                writer = csv.writer(f,dialect=OcgDialect)
                writer.writerow(self._upper_(v['headers']))
                for row in v['it']:
                    writer.writerow(row)

    def _write_(self):
        ## init the value file
        f_value = self._get_file_object_('value')
        try:
            build = True
            for coll in self:
                if build:
                    kit = KeyedIterator(coll)
                    f_writer = csv.writer(f_value,dialect=OcgDialect)
                    f_writer.writerow(kit.get_headers(upper=True))
                    
                    ## write request level identifier files ####################
                    rits = kit.get_request_iters()
                    self._write_iter_dict_(rits)
                    ############################################################
                                
                    build = False
                for row in kit.iter_list(coll):
                    f_writer.writerow(row)  
        finally:  
            f_value.close()
        
        ## write dimension identifiers #########################################
        self._write_iter_dict_(kit.get_dimension_iters())
        
        ## write the shape idx #################################################
        dct = {'projection':coll.projection,'data':kit.gid}
#        dct = {'projection':coll.projection,'data':kit.gid.storage}
        shpidx_path = os.path.join(self.wd,self.prefix+'_'+'shp')
        os.mkdir(shpidx_path)
        shpidx = ShpIdxIdentifierConverter([dct],prefix=(self.prefix+'_shpidx'),wd=shpidx_path,
                                           add_meta=False,nest=False)
        shpidx.write()
        ########################################################################
        
        return(self.wd)
    
    def _get_file_object_(self,prefix):
        return(open(self._get_path_(prefix),'w'))
    
    def _get_path_(self,prefix):
        return(os.path.join(self.wd,self.prefix+'_'+prefix+'.csv'))
    
    def _upper_(self,seq):
        return([s.upper() for s in seq])