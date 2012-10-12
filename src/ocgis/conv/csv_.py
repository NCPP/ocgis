import csv
from ocgis.conv.converter import OcgConverter
from csv import excel


class OcgDialect(excel):
    lineterminator = '\n'
    
class CsvConverter(OcgConverter):
    _ext = 'csv'
    
    def __init__(self,*args,**kwds):
#        self.wkt = kwds.pop('wkt')
#        self.wkb = kwds.pop('wkb')
        
        super(CsvConverter,self).__init__(*args,**kwds)
        
#    def _get_headers_raw_(self):
#        raise(NotImplementedError)
#    
#    def _get_headers_attr_(self):
#        raise(NotImplementedError)
#    
#    def _write_(self,headers):
#        raise(NotImplementedError)
    
    def write(self):
        path = self.get_path()
        build = True
        with open(path,'w') as f:
            writer = csv.writer(f,dialect=OcgDialect)
            for coll,geom_dict in self:
                if build:
                    headers = self.get_headers(upper=True)
                    writer.writerow(headers)
                    build = False
                for row,geom in self.get_iter(coll):
#                    row.pop()
#                    if self.wkt or self.wkb:
#                        raise(NotImplementedError)
                    writer.writerow(row)
        return(path)