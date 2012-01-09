from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
import zipfile
import io


class SqliteConverter(OcgConverter):
    
    def _convert_(self):
        url = self.db.metadata.bind.url.database
        return(url)
        
    def _response_(self,payload):
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        zip.write(payload,arcname=self.base_name+'.sqlite')
        self.write_meta(zip)
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        return(zip_stream)