import netCDF4 as nc

import datetime
from ocgis.api.request.driver.vector import DriverVector
from ocgis.calc.base import AbstractMultivariateFunction, AbstractKeyedOutputFunction
import ocgis
from ocgis.api.parms.definition import OutputFormat
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.calc.eval_function import MultivariateEvalFunction
from ocgis.conv.base import AbstractConverter
from ocgis import constants
from ocgis.exc import DefinitionValidationError
from ocgis.interface.base.crs import CFWGS84


class NcConverter(AbstractConverter):
    _ext = 'nc'

    @classmethod
    def validate_ops(cls, ops):

        def _raise_(msg, ocg_arugument=OutputFormat):
            raise DefinitionValidationError(ocg_arugument, msg)

        # we can only write one requestdataset to netCDF
        if len(ops.dataset) > 1 and ops.calc is None:
            msg = ('Data packages (i.e. more than one RequestDataset) may not be written to netCDF. '
                   'There are currently {dcount} RequestDatasets. Note, this is different than a '
                   'multifile dataset.'.format(dcount=len(ops.dataset)))
            _raise_(msg, OutputFormat)
        # we can write multivariate functions to netCDF however
        else:
            if ops.calc is not None and len(ops.dataset) > 1:
                # count the occurrences of these classes in the calculation list.
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
                    # there is a multivariate calculation and this requires multiple request dataset
                    pass

        # clipped data which creates an arbitrary geometry may not be written to netCDF
        if ops.spatial_operation != 'intersects':
            msg = 'Only "intersects" spatial operation allowed for netCDF output. Arbitrary geometries may not currently be written.'
            _raise_(msg, OutputFormat)
        # data may not be aggregated either
        if ops.aggregate:
            msg = 'Data may not be aggregated for netCDF output. The aggregate parameter must be False.'
            _raise_(msg, OutputFormat)
        # either the input data CRS or WGS84 is required for data output
        if ops.output_crs is not None and not isinstance(ops.output_crs, CFWGS84):
            msg = 'CFWGS84 is the only acceptable overloaded output CRS at this time for netCDF output.'
            _raise_(msg, OutputFormat)
        # calculations on raw values are not relevant as not aggregation can occur anyway.
        if ops.calc is not None:
            if ops.calc_raw:
                msg = 'Calculations must be performed on original values (i.e. calc_raw=False) for netCDF output.'
                _raise_(msg)
            # no keyed output functions to netCDF
            if OcgCalculationEngine._check_calculation_members_(ops.calc, AbstractKeyedOutputFunction):
                msg = 'Keyed function output may not be written to netCDF.'
                _raise_(msg)

    def _finalize_(self, ds):
        ds.close()

    def _build_(self, coll):
        ds = nc.Dataset(self.path, 'w', format=self._get_file_format_())
        return ds

    def _get_file_format_(self):
        file_format = set()
        # if no operations are present, use the default data model
        if self.ops is None:
            ret = constants.NETCDF_DEFAULT_DATA_MODEL
        else:
            for rd in self.ops.dataset.iter_request_datasets():
                try:
                    rr = rd.source_metadata['file_format']
                except KeyError:
                    # likely a shapefile request dataset which does not have an origin netcdf data format
                    if isinstance(rd.driver, DriverVector):
                        continue
                    else:
                        raise
                if isinstance(rr, basestring):
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
                    # likely all field objects in the dataset. use the default netcdf data model
                    ret = constants.NETCDF_DEFAULT_DATA_MODEL
        return ret

    @staticmethod
    def _write_archetype_(arch, dataset, is_file_only):
        """
        Write a field to a netCDF dataset object.

        :param arch: The field to write.
        :type arch: :class:`~ocgis.Field`
        :param dataset: An open netCDF4 dataset object.
        :type dataset: :class:`netCDF4.Dataset`
        :param bool file_only: If ``True``, this is writing the template file only and there is no data fill.
        """

        arch.write_to_netcdf_dataset(dataset, file_only=is_file_only)
    
    def _write_coll_(self, ds, coll):
        """
        Write a spatial collection to an open netCDF4 dataset object.

        :param ds: An open dataset object.
        :type ds: :class:`netCDF4.Dataset`
        :param coll: The collection containing data to write.
        :type coll: :class:`~ocgis.SpatialCollection`
        """

        # get the target field from the collection
        arch = coll._archetype_field
        """:type arch: :class:`ocgis.Field`"""

        # get from operations if this is file only.
        try:
            is_file_only = self.ops.file_only
        except AttributeError:
            # no operations object available
            is_file_only = False

        self._write_archetype_(arch, ds, is_file_only)

        # append to the history attribute
        history_str = '\n{dt} UTC ocgis-{release}'.format(dt=datetime.datetime.utcnow(), release=ocgis.__release__)
        if self.ops is not None:
            history_str += ': {0}'.format(self.ops)
        original_history_str = ds.__dict__.get('history', '')
        setattr(ds, 'history', original_history_str+history_str)


class NcUgrid2DFlexibleMeshConverter(NcConverter):

    @classmethod
    def validate_ops(cls, ops):
        NcConverter.validate_ops(ops)
        should_raise = False
        if ops.abstraction == 'point':
            should_raise = True
        else:
            for ds in ops.dataset.itervalues():
                try:
                    abstraction = ds.s_abstraction
                except AttributeError:
                    # might be a field object
                    abstraction = ds.spatial.abstraction
                if abstraction == 'point':
                    should_raise = True
                    break

        if should_raise:
            msg = 'Only polygons may be written to "{0}"'.\
                format(constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH, ops.abstraction)
            raise DefinitionValidationError(OutputFormat, msg)

    @staticmethod
    def _write_archetype_(arch, dataset, is_file_only):
        poly = arch.spatial.geom.polygon
        """:type poly: :class:`ocgis.SpatialGeometryPolygonDimension`"""

        try:
            poly.write_to_netcdf_dataset_ugrid(dataset)
        except AttributeError:
            if poly is None:
                msg = 'Field does not have a polygon representation.'
                raise ValueError(msg)
            else:
                raise