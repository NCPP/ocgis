import datetime

import six

import ocgis
from ocgis import RequestDataset
from ocgis import env
from ocgis.calc.base import AbstractMultivariateFunction, AbstractKeyedOutputFunction
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.calc.eval_function import MultivariateEvalFunction
from ocgis.constants import KeywordArgument
from ocgis.conv.base import AbstractCollectionConverter
from ocgis.driver.nc import DriverNetcdf, DriverNetcdfCF
from ocgis.exc import DefinitionValidationError
from ocgis.variable.crs import CFWGS84


class NcConverter(AbstractCollectionConverter):
    """
    .. note:: Accepts all parameters to :class:`~ocgis.conv.base.AbstractCollectionConverter`.

    :param options: (``=None``) The following options are valid:

    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    | Option                 | Description                                                                                                                            |
    +========================+========================================================================================================================================+
    | data_model             | The netCDF data model: http://unidata.github.io/netcdf4-python/#netCDF4.Dataset.                                                       |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    | variable_kwargs        | Dictionary of keyword parameters to use for netCDF variable creation. See: http://unidata.github.io/netcdf4-python/#netCDF4.Variable.  |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    | unlimited_to_fixedsize | If ``True``, convert the unlimited dimension to fixed size. Only applies to time and level dimensions.                                 |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------+

    >>> options = {'data_model': 'NETCDF4_CLASSIC'}
    >>> options = {'variable_kwargs': {'zlib': True, 'complevel': 4}}

    :type options: str
    """
    _ext = 'nc'

    @property
    def _variable_kwargs(self):
        try:
            ret = self.ops.output_format_options.get('variable_kwargs', {})
        except AttributeError:
            # Likely "ops" or "output_format_options" is None.
            ret = {}
        return ret

    @classmethod
    def validate_ops(cls, ops):
        from ocgis.ops.parms.definition import OutputFormat

        def _raise_(msg, ocg_arugument=OutputFormat):
            raise DefinitionValidationError(ocg_arugument, msg)

        # We can only write one request dataset to netCDF.
        len_ops_dataset = len(list(ops.dataset))
        if len_ops_dataset > 1 and ops.calc is None:
            msg = 'Data packages (i.e. more than one RequestDataset) may not be written to netCDF. There are ' \
                  'currently {dcount} RequestDatasets. Note, this is different than a multifile dataset.'
            msg = msg.format(dcount=len_ops_dataset)
            _raise_(msg, OutputFormat)
        # We can write multivariate functions to netCDF.
        else:
            if ops.calc is not None and len_ops_dataset > 1:
                # Count the occurrences of these classes in the calculation list.
                klasses_to_check = [AbstractMultivariateFunction, MultivariateEvalFunction]
                multivariate_checks = []
                for klass in klasses_to_check:
                    for calc in ops.calc:
                        multivariate_checks.append(issubclass(calc['ref'], klass))
                if sum(multivariate_checks) != 1:
                    msg = ('Data packages (i.e. more than one RequestDataset) may not be written to netCDF. '
                           'There are currently {dcount} RequestDatasets. Note, this is different than a '
                           'multifile dataset.'.format(dcount=len(ops.dataset)))
                    _raise_(msg, OutputFormat)
                else:
                    # There is a multivariate calculation and this requires multiple request datasets.
                    pass

        # Clipped data which creates an arbitrary geometry may not be written to netCDF.
        if ops.spatial_operation != 'intersects':
            msg = ('Only "intersects" spatial operation allowed for netCDF output. Arbitrary geometries may not '
                   'currently be written.')
            _raise_(msg, OutputFormat)
        # Data may not be aggregated either.
        if ops.aggregate:
            msg = 'Data may not be aggregated for netCDF output. The aggregate parameter must be False.'
            _raise_(msg, OutputFormat)
        # Either the input data CRS or WGS84 is required for data output.
        if ops.output_crs is not None and not isinstance(ops.output_crs, CFWGS84):
            msg = 'CFWGS84 is the only acceptable overloaded output CRS at this time for netCDF output.'
            _raise_(msg, OutputFormat)
        # Calculations on raw values are not relevant as not aggregation can occur anyway.
        if ops.calc is not None:
            if ops.calc_raw:
                msg = 'Calculations must be performed on original values (i.e. calc_raw=False) for netCDF output.'
                _raise_(msg)
            # No keyed output functions to netCDF.
            if OcgCalculationEngine._check_calculation_members_(ops.calc, AbstractKeyedOutputFunction):
                msg = 'Keyed function output may not be written to netCDF.'
                _raise_(msg)

    def _build_(self, coll):
        ret = {'path': self.path, 'dataset_kwargs': {'format': self._get_file_format_()},
               'variable_kwargs': self._variable_kwargs, 'driver': DriverNetcdfCF}
        return ret

    def _finalize_(self, ds):
        pass

    def _get_file_format_(self):
        file_format = set()
        # Use the data model passed to the constructor.
        if self.options.get('data_model') is not None:
            ret = self.options['data_model']
        else:
            # If no operations are present, use the default data model.
            if self.ops is None:
                ret = env.NETCDF_FILE_FORMAT
            else:
                # If operations are available, check the request datasets and determine the best format for output.
                for rd in self.ops.dataset:
                    # Only request dataset will have a possible output format.
                    if not isinstance(rd, RequestDataset):
                        continue
                    try:
                        rr = rd.metadata['file_format']
                    except KeyError:
                        # Likely a shapefile request dataset which does not have an origin netcdf data format.
                        if not isinstance(rd.driver, DriverNetcdf):
                            continue
                        else:
                            raise
                    if isinstance(rr, six.string_types):
                        tu = [rr]
                    else:
                        tu = rr
                    file_format.update(tu)
                if len(file_format) > 1:
                    raise ValueError('Multiple file formats found: {0}'.format(file_format))
                else:
                    try:
                        ret = list(file_format)[0]
                    except IndexError:
                        # Likely all field objects in the dataset. Use the default netCDF data model.
                        ret = env.NETCDF_FILE_FORMAT
        return ret

    def _write_archetype_(self, arch, write_kwargs, variable_kwargs):
        """
        Write a field to a netCDF dataset object.

        :param arch: The field to write.
        :type arch: :class:`ocgis.new_interface.field.OcgField`
        :param dict write_kwargs: Dictionary of parameters needed for the write.
        :param dict variable_kwargs: Optional keyword parameters to pass to the creation of netCDF4 variable objects.
         See http://unidata.github.io/netcdf4-python/#netCDF4.Variable.
        """
        # Append to the history attribute.
        history_str = '\n{dt} UTC ocgis-{release}'.format(dt=datetime.datetime.utcnow(), release=ocgis.__release__)
        if self.ops is not None:
            history_str += ': {0}'.format(self.ops)
        original_history_str = arch.attrs.get('history', '')
        arch.attrs['history'] = original_history_str + history_str

        # Pull in dataset and variable keyword arguments.
        unlimited_to_fixedsize = self.options.get(KeywordArgument.UNLIMITED_TO_FIXED_SIZE, False)
        write_kwargs[KeywordArgument.VARIABLE_KWARGS] = variable_kwargs
        variable_kwargs[KeywordArgument.UNLIMITED_TO_FIXED_SIZE] = unlimited_to_fixedsize

        # This is the output path. The driver handles MPI writing.
        path = write_kwargs.pop(KeywordArgument.PATH)

        # Write the field.
        arch.write(path, **write_kwargs)

    def _write_coll_(self, ds, coll):
        """
        Write a spatial collection to an open netCDF4 dataset object.

        :param ds: An open dataset object.
        :type ds: :class:`netCDF4.Dataset`
        :param coll: The collection containing data to write.
        :type coll: :class:`~ocgis.SpatialCollection`
        """

        # Get the target field from the collection.
        arch = coll.archetype_field
        """:type arch: :class:`ocgis.Field`"""

        self._write_archetype_(arch, ds, self._variable_kwargs)
