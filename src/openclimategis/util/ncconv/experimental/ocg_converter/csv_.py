from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
import io
import zipfile
import csv


class CsvConverter(OcgConverter):
#    __headers__ = ['OCGID','GID','TIME','LEVEL','VALUE','AREA_M2','WKT','WKB']
    
    def __init__(self,*args,**kwds):
        self.as_wkt = kwds.pop('as_wkt',False)
        self.as_wkb = kwds.pop('as_wkb',False)
        self.add_area = kwds.pop('add_area',True)

        ## call the superclass
        super(CsvConverter,self).__init__(*args,**kwds)
        
        self.headers = self.get_headers(self.value_table)
        ## need to extract the time as well
        if 'TID' in self.headers:
            self.headers.insert(self.headers.index('TID')+1,'TIME')
        
        codes = [['add_area','AREA_M2'],['as_wkt','WKT'],['as_wkb','WKB']]
        for code in codes:
            if getattr(self,code[0]):
                self.headers.append(code[1])
        
    
    def get_writer(self,buffer,headers=None):
        writer = csv.writer(buffer)
        if headers is None: headers = self.headers
        writer.writerow(headers)
        writer = csv.DictWriter(buffer,headers)
        return(writer)
    
    def _convert_(self):
        buffer = io.BytesIO()
        writer = self.get_writer(buffer)
        for attrs in self.get_iter(self.value_table,self.headers):
            writer.writerow(attrs)
        buffer.flush()
        return(buffer.getvalue())
    
    
class LinkedCsvConverter(CsvConverter):
    
    def __init__(self,*args,**kwds):
        self.tables = kwds.pop('tables',None)
        
        super(LinkedCsvConverter,self).__init__(*args,**kwds)
        
        if self.tables is None and self.use_stat:
            tables = kwds.pop('tables',['Geometry','Stat'])
        elif self.tables is None and not self.use_stat:
            tables = kwds.pop('tables',['Geometry','Time','Value'])
        self.tables = [getattr(self.db,tbl) for tbl in tables]
        
    def _clean_headers_(self,table):
        headers = self.get_headers(table)
        if self.get_tablename(table) == 'geometry':
            codes = [['add_area','AREA_M2'],['as_wkt','WKT'],['as_wkb','WKB']]
            for code in codes:
                if not getattr(self,code[0]):
                    headers.remove(code[1])
        return(headers)
    
    def _convert_(self):
        ## generate the info for writing
        info = []
        for table in self.tables:
            headers = self._clean_headers_(table)
#            headers = self._clean_headers_([h.upper() for h in table.__mapper__.columns.keys()])
            arcname = '{0}_{1}.csv'.format(self.base_name,self.get_tablename(table))
            buffer = io.BytesIO()
            writer = self.get_writer(buffer,headers=headers)
            info.append(dict(headers=headers,
                             writer=writer,
                             arcname=arcname,
                             table=table,
                             buffer=buffer))
        ## write the tables
        for i in info:
            ## loop through each database record
            for attrs in self.get_iter(i['table'],i['headers']):
                i['writer'].writerow(attrs)
            i['buffer'].flush()

        return(info)
    
    def _response_(self,payload):
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        for info in payload:
            zip.writestr(info['arcname'],info['buffer'].getvalue())
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        return(zip_stream)