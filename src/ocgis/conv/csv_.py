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
    
    def write(self):
        path = self.get_path()
        build = True
        with open(path,'w') as f:
            writer = csv.writer(f,dialect=OcgDialect)
            for coll in self:
                if build:
                    headers = self.get_headers(coll)
                    writer.writerow(headers)
                    build = False
                for row,geom in self.get_iter(coll):
                    writer.writerow(row)
        return(path)