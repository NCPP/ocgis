import zipfile
import io
import os


class Zipper(object):
    '''
    >>> base_path = '/tmp/test_zip'
    >>> zip = Zipperbase_pathth)
    >>> zip.get_zip_stream()
    '''
    
    def __init__(self,base_path):
        self.base_path = base_path
    
    def get_zip_stream(self):
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        try:
            items = os.listdir(self.base_path)
            path = self.base_path
        except OSError:
            path,items = os.path.split(self.base_path)
            items = [items]
        for item in items:
            filepath = os.path.join(path,item)
            zip.write(filepath,arcname=item)
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        return(zip_stream)
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()