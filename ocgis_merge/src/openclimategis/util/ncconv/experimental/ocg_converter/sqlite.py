import zipfile
import io
from util.ncconv.experimental.ocg_converter.subocg_converter import SubOcgConverter
from django.conf import settings


class SqliteConverter(SubOcgConverter):
    
    def _convert_(self):
        db = self._true_sub.to_db(to_disk=True,procs=settings.MAXPROCESSES)
        if self.use_stat:
            self.sub.load(db)
        url = db.metadata.bind.url.database
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