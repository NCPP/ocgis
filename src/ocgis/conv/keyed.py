import csv
from ocgis.conv.converter import OcgConverter
from ocgis.conv.csv_ import OcgDialect
from ocgis.util.helpers import get_temp_path
from ocgis.conv.shpidx import ShpIdxConverter
import os.path
import ocgis.api.interp.iocg.dataset.iterators as kits
from collections import namedtuple


class KeyedConverter(OcgConverter):
    _ext = 'csv'
    
    def __init__(self,*args,**kwds):
        super(KeyedConverter,self).__init__(*args,**kwds)
        self.wd = get_temp_path(only_dir=True,wd=self.wd,nest=True)
    
    def write(self):
        ## these variables are here for the shpidx. tricking the iterators at
        ## this time. stripping out unnecessary data to conserve memory.
        DummyColl = namedtuple('DummyColl',['gid','geom'])
        shpidx_cache = []
        ## indicate this is the first time through the iteration.
        build = True
        for coll,geom_dict in self:
            ## save the variable for the shape index writing
            shpidx_cache.append((DummyColl(coll.gid,coll.geom),None))
            ## get the proper iterator for iterator mode
            if self.mode == 'raw':
                its = kits.RawKeyedIterator(coll).get_iters()
            elif self.mode == 'agg':
                its = kits.AggKeyedIterator(coll).get_iters()
            elif self.mode == 'calc':
                its = kits.CalcKeyedIterator(coll).get_iters()
            else:
                raise(NotImplementedError)
            ## perform operations on first iteration
            if build:
                ## make file objects
                files = {}
                for key,value in its.iteritems():
                    path = os.path.join(self.wd,'{0}.csv'.format(key))
                    f = open(path,'w')
                    writer = csv.writer(f,dialect=OcgDialect)
                    writer.writerow(value['headers'])
                    files.update({key:{'w':writer,'f':f}})
            ## write the data
            for key,value in its.iteritems():
                if not build:
                    if key in ['tid','tgid','vid','vlid','cid']:
                        continue
                writer = files[key]['w']
                for row in value['it']():
                    writer.writerow(row)
            build = False
        ## close file objects
        for value in files.itervalues():
            value['f'].close()
        
        ## write the shape index. this generator overloads the standard
        ## generator in the converter.
        def _alt_it_():
            for ii in shpidx_cache:
                yield(ii)
        shpidx = ShpIdxConverter(self.so,base_name='shpidx',use_dir=self.wd,alt_it=_alt_it_)
        shpidx.write()
            
        return(self.wd)
