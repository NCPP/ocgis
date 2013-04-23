import csv
from ocgis.conv.base import OcgConverter
from csv import excel


class OcgDialect(excel):
    lineterminator = '\n'


class CsvConverter(OcgConverter):
    _ext = 'csv'
    
    def _write_(self):
        build = True
        with open(self.path,'w') as f:
            writer = csv.writer(f,dialect=OcgDialect)
            for coll in self:
                if build:
                    headers = coll.get_headers()
                    writer.writerow(headers)
                    build = False
                for geom,row in coll.get_iter():
                    writer.writerow(row)
