import io
import os
import zipfile

from ocgis.constants import OutputFormatName


class Zipper(object):
    def __init__(self, base_path=None):
        self.base_path = base_path

    def get_zip_stream(self):
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED)
        try:
            items = os.listdir(self.base_path)
            path = self.base_path
        except OSError:
            path, items = os.path.split(self.base_path)
            items = [items]
        for item in items:
            filepath = os.path.join(path, item)
            zip.write(filepath, arcname=item)
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        return (zip_stream)

    @classmethod
    def compress_path(cls, path, items):
        zipf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
        for item in items:
            arcname = item.get('arcname')
            zipf.write(item['filename'], arcname=arcname)
        zipf.close()
        return (path)


def get_items(folder):
    items = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if os.path.split(dirpath)[1] == 'shp':
                arcname = os.path.join('shp', filename)
            else:
                arcname = filename
            items.append({'filename': os.path.join(dirpath, filename),
                          'arcname': arcname})
    return (items)


def get_zipped_path(path_zip, folder):
    items = get_items(folder)
    Zipper.compress_path(path_zip, items)
    return (path_zip)


def format_return(ret_path, ops, with_auxiliary_files=False):
    """
    Format an OpenClimateGIS path returning an absolute path to a zip file or simply passing `ret_path` through.
    
    >>> import ocgis
    >>> ops = ocgis.OcgOperations(...)
    >>> ret = ops.execute()
    >>> new_path = ocgis.format_return(ret,ops)
    
    :param ret_path: The path returned by :func:`ocgis.OcgOperations.execute`.
    :type ret_path: str
    :param ops: Instance of :class:`ocgis.OcgOperations`.
    :type ops: :class:`ocgis.OcgOperations`
    :param with_auxiliary_files: If `True`, zip additional contents of CSV and netCDF returns.
    :type with_auxiliary_files: bool
    :raises: NotImplementedError
    :returns: str
    """

    # can do nothing with numpy returns
    if ops.output_format == OutputFormatName.OCGIS:
        raise NotImplementedError
    # the folder containing all output files
    folder = os.path.split(ret_path)[0]
    # name for output zipfile
    path_zip = os.path.join(folder, ops.prefix + '.zip')
    if ops.output_format in ['csv', 'nc']:
        # add all files
        if with_auxiliary_files:
            ret = get_zipped_path(path_zip, folder)
        # only interested in the file.
        else:
            ret = ret_path
    # otherwise return all files
    else:
        ret = get_zipped_path(path_zip, folder)

    return ret
