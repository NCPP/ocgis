import abc
import csv
import fiona
import logging
import os.path
import six
from pprint import pformat

from ocgis import constants, vm
from ocgis import env
from ocgis import messages
from ocgis.base import AbstractOcgisObject
from ocgis.collection.spatial import SpatialCollection
from ocgis.constants import TagName, MPIWriteMode, KeywordArgument, SubcommName
from ocgis.driver.vector import DriverVector
from ocgis.util.helpers import get_iter, get_tuple
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.vmachine.mpi import MPI_RANK

FIONA_FIELD_TYPES_REVERSED = {v: k for k, v in fiona.FIELD_TYPES_MAP.items()}
FIONA_FIELD_TYPES_REVERSED[str] = 'str'


@six.add_metaclass(abc.ABCMeta)
class AbstractConverter(AbstractOcgisObject):
    """
    Base class for all converter objects.
    """

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


@six.add_metaclass(abc.ABCMeta)
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


@six.add_metaclass(abc.ABCMeta)
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
    _add_did_file = True  # Add a descriptor file for the request datasets.
    _add_ugeom = False  # Add user geometry in the output folder.
    _add_ugeom_nest = True  # Nest the user geometry in a overview geometry shapefile folder.
    _add_source_meta = True  # Add a source metadata file.
    _use_upper_keys = True  # If headers should be capitalized.
    _allow_empty = False

    def __init__(self, colls, **kwargs):
        self.colls = colls
        self.ops = kwargs.pop('ops', None)
        self.add_meta = kwargs.pop('add_meta', True)
        self.add_auxiliary_files = kwargs.pop('add_auxiliary_files', True)

        super(AbstractCollectionConverter, self).__init__(**kwargs)

    def __iter__(self):
        for coll in self.colls:
            assert isinstance(coll, SpatialCollection)
            if not self._allow_empty and coll.is_empty:
                continue
            yield coll

    @property
    def geom_uid(self):
        none_target = self.ops
        if none_target is not None:
            none_target = self.ops.geom_uid
        if none_target is None:
            ret = env.DEFAULT_GEOM_UID
        else:
            ret = none_target
        return ret

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

    def _get_or_create_shp_folder_(self):
        path = os.path.join(self.outdir, 'shp')
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def _preformatting_(self, i, coll):
        """Prepare collection before it is written to disk."""
        return coll

    def write(self):
        ocgis_lh('starting write method', self._log, logging.DEBUG)

        # Indicates if user geometries should be written to file.
        write_ugeom = False

        # Path to the output object.
        f = {KeywordArgument.PATH: self.path}

        build = True
        for i, coll in enumerate(self):
            # This will be changed to "write" if we are on the build loop.
            write_mode = MPIWriteMode.APPEND

            if build:
                # During a build loop, create the file and write the first series of records. Let the drivers determine
                # the appropriate write modes for handling parallelism.
                write_mode = None

                # Write the user geometries if selected and there is one present on the incoming collection.
                if self._add_ugeom and coll.has_container_geometries:
                    write_ugeom = True

                if write_ugeom:
                    if vm.rank == 0:
                        # The output file name for the user geometries.
                        ugid_shp_name = self.prefix + '_ugid.shp'
                        if self._add_ugeom_nest:
                            ugeom_fiona_path = os.path.join(self._get_or_create_shp_folder_(), ugid_shp_name)
                        else:
                            ugeom_fiona_path = os.path.join(self.outdir, ugid_shp_name)
                    else:
                        ugeom_fiona_path = None

                build = False

            f[KeywordArgument.WRITE_MODE] = write_mode

            self._write_coll_(f, self._preformatting_(i, coll))

            if write_ugeom:
                with vm.scoped(SubcommName.UGEOM_WRITE, [0]):
                    if not vm.is_null:
                        for subset_field in list(coll.children.values()):
                            subset_field.write(ugeom_fiona_path, write_mode=write_mode, driver=DriverVector)

        # The metadata and dataset descriptor files may only be written if OCGIS operations are present.
        ops = self.ops
        if ops is not None and self.add_auxiliary_files and MPI_RANK == 0:
            # Add OCGIS metadata output if requested.
            if self.add_meta:
                ocgis_lh('adding OCGIS metadata file', 'conv', logging.DEBUG)
                from ocgis.conv.meta import MetaOCGISConverter

                lines = MetaOCGISConverter(ops).write()
                out_path = os.path.join(self.outdir, self.prefix + '_' + MetaOCGISConverter._meta_filename)
                with open(out_path, 'w') as f:
                    f.write(lines)

            # Add the dataset descriptor file if requested.
            if self._add_did_file:
                ocgis_lh('writing dataset description (DID) file', 'conv', logging.DEBUG)
                path = os.path.join(self.outdir, self.prefix + '_did.csv')
                _write_dataset_identifier_file_(path, ops)

            # Add source metadata if requested.
            if self._add_source_meta:
                ocgis_lh('writing source metadata file', 'conv', logging.DEBUG)
                path = os.path.join(self.outdir, self.prefix + '_source_metadata.txt')
                _write_source_meta_(path, ops)

        # Return the internal path unless overloaded by subclasses.
        ret = self._get_return_()

        return ret


@six.add_metaclass(abc.ABCMeta)
class AbstractTabularConverter(AbstractCollectionConverter):
    """
    .. note:: Accepts all parameters to :class:`~ocgis.conv.base.AbstractFileConverter`.

    :keyword bool melted: (``=False``) If ``True``, use a melted tabular output format with variable values collected in
     a single column.
    """

    def __init__(self, *args, **kwargs):
        self.melted = kwargs.pop(KeywordArgument.MELTED, None) or False
        super(AbstractTabularConverter, self).__init__(*args, **kwargs)


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
    from ocgis.conv.nc import NcConverter
    from ocgis.conv.meta import MetaOCGISConverter, MetaJSONConverter

    mmap = {constants.OutputFormatName.SHAPEFILE: ShpConverter,
            constants.OutputFormatName.CSV: CsvConverter,
            constants.OutputFormatName.CSV_SHAPEFILE: CsvShapefileConverter,
            constants.OutputFormatName.OCGIS: NumpyConverter,
            constants.OutputFormatName.GEOJSON: GeoJsonConverter,
            constants.OutputFormatName.NETCDF: NcConverter,
            constants.OutputFormatName.METADATA_JSON: MetaJSONConverter,
            constants.OutputFormatName.METADATA_OCGIS: MetaOCGISConverter,
            }

    # ESMF is an optional dependendency.
    if env.USE_ESMF:
        from ocgis.conv.esmpy import ESMPyConverter
        mmap[constants.OutputFormatName.ESMPY_GRID] = ESMPyConverter

    return mmap


def _write_dataset_identifier_file_(path, ops):
    from ocgis.conv.csv_ import OcgDialect
    rows = []
    headers = ['DID', 'VARIABLE', 'STANDARD_NAME', 'LONG_NAME', 'UNITS', 'URI', 'GROUP']
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, headers, dialect=OcgDialect)
        writer.writeheader()
        # writer.writerow(headers)
        for element in ops.dataset:
            row_template = {'DID': element.uid}
            if element.has_data_variables:
                try:
                    itr = get_iter(element.variable)
                except AttributeError:
                    itr = element.get_by_tag(TagName.DATA_VARIABLES)
                for idx, variable in enumerate(itr):
                    row = row_template.copy()
                    try:
                        attrs = variable.attrs
                        units = variable.units
                        group = variable.group
                        uri = None
                        variable_name = variable.name
                    except AttributeError:
                        attrs = element.metadata['variables'][variable]['attrs']
                        units = get_tuple(element.units)[idx]
                        group = None
                        uri = element.uri
                        variable_name = variable
                    row['STANDARD_NAME'] = attrs.get('standard_name')
                    row['LONG_NAME'] = attrs.get('long_name')
                    row['UNITS'] = units
                    row['GROUP'] = group
                    row['URI'] = uri
                    row['VARIABLE'] = variable_name
                    rows.append(row)
        writer.writerows(rows)


def _write_source_meta_(path, operations):
    to_write = []
    for ctr, element in enumerate(operations.dataset):
        to_write.append('===========================')
        to_write.append('== Dataset Identifer (DID): {}'.format(element.uid))
        to_write.append('===========================')
        to_write.append('')
        dimension_map = pformat(element.dimension_map.as_dict())
        to_write.append('== Dimension Map ==========')
        to_write.append('')
        to_write.append(dimension_map)
        try:
            metadata = element.driver.get_dump_report()
        except TypeError:
            # TODO: Collections/fields should be able to generate metadata.
            # Field objects cannot create dump reports.
            pass
        else:
            to_write.append('')
            to_write.append('== Metadata Dump ==========')
            to_write.append('')
            to_write.extend(metadata)
        to_write.append('')
    with open(path, 'w') as f:
        for line in to_write:
            f.write(line)
            f.write('\n')
