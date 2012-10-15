import zipfile
import io
import os


class Zipper(object):
    '''
    >>> base_dir = '/tmp/test_zip'
    >>> zip = Zipper(base_dir)
    >>> zip.get_zip_stream()
    '''
    
    def __init__(self,base_dir):
        self.base_dir = base_dir
    
    def get_zip_stream(self):
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        for item in os.listdir(self.base_dir):
            filepath = os.path.join(self.base_dir,item)
            zip.write(filepath,arcname=item)
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        return(zip_stream)
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()