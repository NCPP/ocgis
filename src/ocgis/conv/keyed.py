import csv
from ocgis.conv.converter import OcgConverter
from ocgis.conv.csv_ import OcgDialect
from ocgis.util.helpers import get_temp_path
from ocgis.conv.shpidx import ShpIdxConverter
import os.path
from ocgis.api.interp.iocg.dataset.iterators import RawKeyedIterator,\
    AggKeyedIterator, CalcKeyedIterator


class KeyedConverter(OcgConverter):
    _ext = 'csv'
    
    def __init__(self,*args,**kwds):
#        self.wkt = kwds.pop('wkt')
#        self.wkb = kwds.pop('wkb')
        
        super(KeyedConverter,self).__init__(*args,**kwds)
        
        self.wd = get_temp_path(only_dir=True,wd=self.wd,nest=True)
        
#    def _get_headers_raw_(self):
#        raise(NotImplementedError)
#    
#    def _get_headers_attr_(self):
#        raise(NotImplementedError)
#    
#    def _write_(self,headers):
#        raise(NotImplementedError)

    
    def write(self):
        ## write the shape index
        shpidx = ShpIdxConverter(self.so,base_name='shpidx',use_dir=self.wd)
        shpidx.write()
        
#        value_path = os.path.join(self.wd,'value.csv')
#        value_file = open(value_path,'w')
#        value_writer = csv.writer(value_file,dialect=OcgDialect)
#        value_writer.writerow(['GID','TID','VID','VLID','VALUE'])
        
        build = True
        for coll,geom_dict in self:
            if self.mode == 'raw':
                its = RawKeyedIterator(coll).get_iters()
            elif self.mode == 'agg':
                its = AggKeyedIterator(coll).get_iters()
            elif self.mode == 'calc':
                its = CalcKeyedIterator(coll).get_iters()
            else:
                raise(NotImplementedError)
            if build:
                ## make file objects
                files = {}
                for key,value in its.iteritems():
                    path = os.path.join(self.wd,'{0}.csv'.format(key))
                    f = open(path,'w')
                    writer = csv.writer(f,dialect=OcgDialect)
                    writer.writerow(value['headers'])
                    files.update({key:{'w':writer,'f':f}})
#                import ipdb;ipdb.set_trace()
#                ## write the static files
#                for key,value in its.iteritems():
#                    path = os.path.join(self.wd,'{0}.csv'.format(key))
#                    with open(path,'w') as f:
#                        writer = csv.writer(f,dialect=OcgDialect)
#                        writer.writerow([h.upper() for h in value['headers']])
#                        for row in value['it']:
#                            writer.writerow(row)
#                build = False
            for key,value in its.iteritems():
                if not build:
                    if key in ['tid','tgid','vid','vlid','cid']:
                        continue
                writer = files[key]['w']
                for row in value['it']():
                    writer.writerow(row)
            build = False
                        
        for value in files.itervalues():
            value['f'].close()
            
            
            
#        path = self.get_path()
#        with open(path,'w') as f:
#            writer = csv.writer(f,dialect=OcgDialect)
#            for ii,(coll,geom_dict) in enumerate(self):
##                print 'converter loop:',ii
#                if ii == 0:
#                    headers = self.get_headers(coll)
#                    writer.writerow(headers)
#                for row,geom in self.get_iter(coll):
##                    row.pop()
##                    if self.wkt or self.wkb:
##                        raise(NotImplementedError)
#                    writer.writerow(row)
        return(self.wd)
    
#    def get_path(self):
    
#    def _write_component_(self,base_name,it):


class KeyedIterators(object):
    
    def __init__(self,coll):
        self.coll = coll
        
    
        
    