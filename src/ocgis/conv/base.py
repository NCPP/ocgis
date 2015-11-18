from collections import OrderedDict
import os.path
import abc
import csv
import logging

import numpy as np
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
import fiona

from ocgis import messages
from ocgis import constants
from ocgis.api.request.driver.vector import DriverVector
from ocgis.interface.base.field import Field
from ocgis.util.inspect import Inspect
from ocgis.util.logging_ocgis import ocgis_lh

FIONA_FIELD_TYPES_REVERSED = {v: k for k, v in fiona.FIELD_TYPES_MAP.iteritems()}
FIONA_FIELD_TYPES_REVERSED[str] = 'str'


class AbstractConverter(object):
    """
    Base class for all converter objects.
    """

    __metaclass__ = abc.ABCMeta

    @classmethod
    def validate_ops(cls, ops):
        """
        Validate an operations object.

        :param ops: The input operations object to validate.
        :type ops: :class:`ocgis.OcgOperations`
        :raises: DefinitionValidationError
        """

    @abc.abstractmethod
    def write(self):
        """
        Write the output. This may be to file, stream, etc.
        """


class AbstractFileConverter(AbstractConverter):
    """
    Base class for all converters writing to file.

    .. note:: Accepts all parameters to :class:`ocgis.conv.base.AbstractConverter`.

    :param prefix: The string prepended to the output file or directory.
    :type prefix: str
    :param outdir: Path to the output directory.
    :type outdir: str
    :param overwrite: (``=False``) If ``True``, attempt to overwrite any existing output files.
    :type overwrite: bool
    :param options: (``=None``) A dictionary of converter-specific options. See converters for options documentation.
    :type options: dict
    """

    __metaclass__ = abc.ABCMeta
    _ext = None

    def __init__(self, prefix=None, outdir=None, overwrite=False, options=None):
        self.outdir = outdir
        self.overwrite = overwrite
        self.prefix = prefix

        if options is None:
            self.options = {}
        else:
            self.options = options

        if self._ext is None:
            self.path = self.outdir
        else:
            self.path = os.path.join(self.outdir, prefix + '.' + self._ext)
            if os.path.exists(self.path):
                if not self.overwrite:
                    msg = messages.M3.format(self.path)
                    raise IOError(msg)

        self._log = ocgis_lh.get_logger('conv')


class AbstractCollectionConverter(AbstractFileConverter):
    """
    Base converter object for convert sequences of collections.

    .. note:: Accepts all parameters to :class:`ocgis.conv.base.AbstractFileConverter`.

    :param colls: A sequence of :class:`~ocgis.SpatialCollection` objects.
    :type colls: sequence of :class:`~ocgis.SpatialCollection`
    :param ops: (``=None``) Optional operations definition. This is required for some converters.
    :type ops: :class:`~ocgis.OcgOperations`
    :param add_meta: (``=True``) If ``False``, do not add a source and OpenClimateGIS metadata file.
    :type add_meta: bool
    :param add_auxiliary_files: (``=True``) If ``False``, do not create an output folder. Write only the target ouput file.
    :type add_auxiliary_files: bool
    """

    __metaclass__ = abc.ABCMeta
    _add_did_file = True  # Add a descriptor file for the request datasets.
    _add_ugeom = False  # Add user geometry in the output folder.
    _add_ugeom_nest = True  # Nest the user geometry in a overview geometry shapefile folder.
    _add_source_meta = True  # Add a source metadata file.
    _use_upper_keys = True  # If headers should be capitalized.

    def __init__(self, colls, **kwargs):
        self.colls = colls
        self.ops = kwargs.pop('ops', None)
        self.add_meta = kwargs.pop('add_meta', True)
        self.add_auxiliary_files = kwargs.pop('add_auxiliary_files', True)

        super(AbstractCollectionConverter, self).__init__(**kwargs)

    def get_headers(self, coll):
        """
        :type coll: :class:`ocgis.SpatialCollection`
        :returns: A list of headers from the first element return from the collection iterator.
        :rtype: [str, ...]
        """

        ret = self.get_iter_from_spatial_collection(coll)
        ret = ret.next()
        ret = ret[1].keys()
        return ret

    def get_iter_from_spatial_collection(self, coll):
        """
        :type coll: :class:`ocgis.SpatialCollection`
        :returns: A generator from the input collection.
        :rtype: generator
        """

        itr = coll.get_iter_dict(use_upper_keys=self._use_upper_keys)
        return itr

    def _build_(self, *args, **kwargs):
        raise NotImplementedError

    def _clean_outdir_(self):
        """
        Remove previous output file from :attr:`ocgis.conv.base.AbstractFileConverter`.
        """

    def _get_return_(self):
        return self.path

    def _write_coll_(self, f, coll):
        raise NotImplementedError

    def _finalize_(self, *args, **kwargs):
        raise NotImplementedError

    def _get_or_create_shp_folder_(self):
        path = os.path.join(self.outdir, 'shp')
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @staticmethod
    def _get_should_append_to_unique_geometry_store_(store, geom, ugid):
        """
        :param sequence store:
        :param geom:
        :type geom: :class:`shapely.Geometry`
        :param ugid:
        :type ugid: int
        """

        ret = True
        test_all = []
        for row in store:
            test_geom = row['geom'].almost_equals(geom)
            test_ugid = row['ugid'] == ugid
            test_all.append(all([test_geom, test_ugid]))
        if any(test_all):
            ret = False
        return ret

    def write(self):
        ocgis_lh('starting write method', self._log, logging.DEBUG)

        unique_geometry_store = []

        # indicates if user geometries should be written to file
        write_ugeom = False

        try:
            build = True

            for coll in iter(self.colls):
                if build:

                    # write the user geometries if configured and there is one present on the incoming collection.
                    if self._add_ugeom and coll.geoms.values()[0] is not None:
                        write_ugeom = True

                    f = self._build_(coll)
                    if write_ugeom:
                        ugid_shp_name = self.prefix + '_ugid.shp'
                        ugid_csv_name = self.prefix + '_ugid.csv'

                        if self._add_ugeom_nest:
                            fiona_path = os.path.join(self._get_or_create_shp_folder_(), ugid_shp_name)
                        else:
                            fiona_path = os.path.join(self.outdir, ugid_shp_name)

                        if coll.meta is None:
                            # convert the collection properties to fiona properties
                            from fiona_ import AbstractFionaConverter

                            fiona_properties = get_schema_from_numpy_dtype(coll.properties.values()[0].dtype)

                            fiona_schema = {'geometry': 'MultiPolygon', 'properties': fiona_properties}
                            fiona_meta = {'schema': fiona_schema, 'driver': 'ESRI Shapefile'}
                        else:
                            fiona_meta = coll.meta

                        # always use the CRS from the collection. shapefile metadata will always be WGS84, but it may be
                        # overloaded in the operations.
                        fiona_meta['crs'] = coll.crs.value

                        # selection geometries will always come out as MultiPolygon regardless if they began as points.
                        # points are buffered during the subsetting process.
                        fiona_meta['schema']['geometry'] = 'MultiPolygon'

                        fiona_object = fiona.open(fiona_path, 'w', **fiona_meta)

                    build = False
                self._write_coll_(f, coll)
                if write_ugeom:
                    # write the overview geometries to disk
                    r_geom = coll.geoms.iteritems().next()
                    uid_value = r_geom[0]
                    r_geom = r_geom[1]
                    if isinstance(r_geom, Polygon):
                        r_geom = MultiPolygon([r_geom])
                    # see if this geometry is in the unique geometry store
                    should_append = self._get_should_append_to_unique_geometry_store_(unique_geometry_store, r_geom,
                                                                                      uid_value)
                    if should_append:
                        unique_geometry_store.append({'geom': r_geom, 'ugid': uid_value})

                        # if it is unique write the geometry to the output files
                        coll.write_ugeom(fobject=fiona_object)
        finally:

            # errors are masked if the processing failed and file objects, etc. were not properly created. if there are
            # UnboundLocalErrors pass them through to capture the error that lead to the objects not being created.

            try:
                try:
                    self._finalize_(f)
                except UnboundLocalError:
                    pass
            except Exception as e:
                # this the exception we want to log
                ocgis_lh(exc=e, logger=self._log)
            finally:
                if write_ugeom:
                    try:
                        fiona_object.close()
                    except UnboundLocalError:
                        pass

        # the metadata and dataset descriptor files may only be written if OCGIS operations are present.
        if self.ops is not None and self.add_auxiliary_files == True:
            # added OCGIS metadata output if requested.
            if self.add_meta:
                ocgis_lh('adding OCGIS metadata file', 'conv', logging.DEBUG)
                from ocgis.conv.meta import MetaOCGISConverter

                lines = MetaOCGISConverter(self.ops).write()
                out_path = os.path.join(self.outdir, self.prefix + '_' + MetaOCGISConverter._meta_filename)
                with open(out_path, 'w') as f:
                    f.write(lines)

            # add the dataset descriptor file if requested
            if self._add_did_file:
                ocgis_lh('writing dataset description (DID) file', 'conv', logging.DEBUG)
                from ocgis.conv.csv_ import OcgDialect

                headers = ['DID', 'VARIABLE', 'ALIAS', 'URI', 'STANDARD_NAME', 'UNITS', 'LONG_NAME']
                out_path = os.path.join(self.outdir, self.prefix + '_did.csv')
                with open(out_path, 'w') as f:
                    writer = csv.writer(f, dialect=OcgDialect)
                    writer.writerow(headers)
                    for rd in self.ops.dataset.itervalues():
                        try:
                            for d in rd:
                                row = [rd.did, d['variable'], d['alias'], rd.uri]
                                try:
                                    ref_variable = rd.source_metadata['variables'][d['variable']]['attrs']
                                except KeyError:
                                    if isinstance(rd.driver, DriverVector):
                                        # not be present in metadata
                                        ref_variable = {}
                                    else:
                                        raise
                                row.append(ref_variable.get('standard_name', None))
                                row.append(ref_variable.get('units', None))
                                row.append(ref_variable.get('long_name', None))
                                writer.writerow(row)
                        except NotImplementedError:
                            if isinstance(rd, Field):
                                for variable in rd.variables.itervalues():
                                    row = [rd.uid, variable.name, variable.alias, None,
                                           variable.attrs.get('standard_name'), variable.units,
                                           variable.attrs.get('long_name')]
                                    writer.writerow(row)
                            else:
                                raise

            # add source metadata if requested
            if self._add_source_meta:
                ocgis_lh('writing source metadata file', 'conv', logging.DEBUG)
                out_path = os.path.join(self.outdir, self.prefix + '_source_metadata.txt')
                to_write = []

                for rd in self.ops.dataset.iter_request_datasets():
                    ip = Inspect(request_dataset=rd)
                    to_write += ip.get_report_possible()

                with open(out_path, 'w') as f:
                    f.writelines(Inspect.newline.join(to_write))

        # return the internal path unless overloaded by subclasses.
        ret = self._get_return_()

        return ret


class AbstractTabularConverter(AbstractCollectionConverter):
    """
    .. note:: Accepts all parameters to :class:`~ocgis.conv.base.AbstractFileConverter`.

    :keyword bool melted: (``=False``) If ``True``, use a melted tabular output format with variable values collected in
     a single column.
    """

    def __init__(self, *args, **kwargs):
        self.melted = kwargs.pop('melted', None) or False
        super(AbstractTabularConverter, self).__init__(*args, **kwargs)

    def get_iter_from_spatial_collection(self, coll):
        """
        :type coll: :class:`ocgis.SpatialCollection`
        :returns: A generator from the input collection.
        :rtype: generator
        """

        itr = coll.get_iter_dict(use_upper_keys=self._use_upper_keys, melted=self.melted)
        return itr


def get_converter(output_format):
    """
    Return the converter based on output extensions or key.

    :param output_format: The target output format for conversion.
    :type output_format: str
    :rtype: :class:`ocgis.conv.base.AbstractConverter`
    """

    return get_converter_map()[output_format]


def get_converter_map():
    """
    :returns: A dictionary with keys corresponding to an output format's short name. Values correspond to the converter
     class.
    :rtype: dict
    """

    from ocgis.conv.fiona_ import ShpConverter, GeoJsonConverter
    from ocgis.conv.csv_ import CsvConverter, CsvShapefileConverter
    from ocgis.conv.numpy_ import NumpyConverter
    from ocgis.conv.nc import NcConverter, NcUgrid2DFlexibleMeshConverter
    from ocgis.conv.meta import MetaOCGISConverter, MetaJSONConverter
    # from ocgis.conv.esmpy import ESMPyConverter

    mmap = {constants.OUTPUT_FORMAT_SHAPEFILE: ShpConverter,
            constants.OUTPUT_FORMAT_CSV: CsvConverter,
            constants.OUTPUT_FORMAT_CSV_SHAPEFILE: CsvShapefileConverter,
            constants.OUTPUT_FORMAT_NUMPY: NumpyConverter,
            constants.OUTPUT_FORMAT_GEOJSON: GeoJsonConverter,
            constants.OUTPUT_FORMAT_NETCDF: NcConverter,
            constants.OUTPUT_FORMAT_METADATA_JSON: MetaJSONConverter,
            constants.OUTPUT_FORMAT_METADATA_OCGIS: MetaOCGISConverter,
            constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH: NcUgrid2DFlexibleMeshConverter,
            # constants.OUTPUT_FORMAT_ESMPY_GRID: ESMPyConverter
            }

    return mmap


def get_schema_from_numpy_dtype(dtype):
    ret = OrderedDict()
    for name in dtype.names:
        name_dtype, _ = dtype.fields[name]
        if name_dtype.str.startswith('|S'):
            ftype = 'str:{0}'.format(name_dtype.itemsize)
        else:
            ftype = type(np.array(0, dtype=name_dtype).item())
            ftype = FIONA_FIELD_TYPES_REVERSED[ftype]
        ret[name] = ftype
    return ret
