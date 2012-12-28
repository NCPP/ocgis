import csv
from ocgis.conv.converter import OcgConverter
from ocgis.conv.csv_ import OcgDialect
from ocgis.util.helpers import get_temp_path
from ocgis.conv.shpidx import ShpIdxIdentifierConverter
import os.path
from ocgis.api.dataset.collection.iterators import KeyedIterator


class KeyedConverter(OcgConverter):
    _ext = 'csv'
    
    def __init__(self,*args,**kwds):
        super(KeyedConverter,self).__init__(*args,**kwds)
        self.wd = get_temp_path(only_dir=True,wd=self.wd,nest=True)
    
    def _write_iter_dict_(self,dct):
        for k,v in dct.iteritems():
            with open(self._get_path_(k),'w') as f:
                writer = csv.writer(f,dialect=OcgDialect)
                writer.writerow(self._upper_(v['headers']))
                for row in v['it']:
                    writer.writerow(row)

    def write(self):
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
        dct = {'projection':coll.projection,'data':kit.gid.storage}
        shpidx_path = os.path.join(self.wd,'shp')
        os.mkdir(shpidx_path)
        shpidx = ShpIdxIdentifierConverter([dct],base_name='shpidx',use_dir=shpidx_path)
        shpidx.write()
        ########################################################################
        
        return(self.wd)
#        
#        import ipdb;ipdb.set_trace()
#        ## these variables are here for the shpidx. tricking the iterators at
#        ## this time. stripping out unnecessary data to conserve memory.
#        DummyColl = namedtuple('DummyColl',['gid','geom'])
#        shpidx_cache = []
#        ## indicate this is the first time through the iteration.
#        build = True
#        for coll,geom_dict in self:
#            ## save the variable for the shape index writing
#            shpidx_cache.append((DummyColl(coll.gid,coll.geom),None))
#            ## get the proper iterator for iterator mode
#            raise(NotImplementedError)
#            if self.mode == 'raw':
#                its = kits.RawKeyedIterator(coll).get_iters()
#            elif self.mode == 'agg':
#                its = kits.AggKeyedIterator(coll).get_iters()
#            elif self.mode == 'calc':
#                its = kits.CalcKeyedIterator(coll).get_iters()
#            elif self.mode == 'multi':
#                its = kits.MultiKeyedIterator(coll).get_iters()
#            ## perform operations on first iteration
#            if build:
#                ## make file objects
#                files = {}
#                for key,value in its.iteritems():
#                    path = os.path.join(self.wd,'{0}.csv'.format(key))
#                    f = open(path,'w')
#                    writer = csv.writer(f,dialect=OcgDialect)
#                    writer.writerow(value['headers'])
#                    files.update({key:{'w':writer,'f':f}})
#            ## write the data
#            for key,value in its.iteritems():
#                if not build:
#                    if key in ['tid','tgid','vid','vlid','cid']:
#                        continue
#                writer = files[key]['w']
#                for row in value['it']():
#                    writer.writerow(row)
#            build = False
#        ## close file objects
#        for value in files.itervalues():
#            value['f'].close()
#        
#        ## write the shape index. this generator overloads the standard
#        ## generator in the converter.
#        def _alt_it_():
#            for ii in shpidx_cache:
#                yield(ii)
#        shpidx = ShpIdxConverter(self.so,base_name='shpidx',use_dir=self.wd,alt_it=_alt_it_)
#        shpidx.write()
#            
#        return(self.wd)
    
    def _get_file_object_(self,prefix):
        return(open(self._get_path_(prefix),'w'))
    
    def _get_path_(self,prefix):
        return(os.path.join(self.wd,prefix+'.csv'))
    
    def _upper_(self,seq):
        return([s.upper() for s in seq])