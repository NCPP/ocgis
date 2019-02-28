import datetime
import itertools
from collections import OrderedDict
from copy import deepcopy

import fiona
import numpy as np
from ocgis import constants, vm
from ocgis.constants import MPIWriteMode, DimensionName, KeywordArgument, DriverKey, DMK, SourceIndexType, VariableName
from ocgis.driver.base import driver_scope, AbstractTabularDriver
from ocgis.driver.dimension_map import DimensionMap
from ocgis.environment import get_dtype
from ocgis.exc import RequestableFeature
from ocgis.util.helpers import is_auto_dtype
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.geom import GeometryVariable
from shapely.geometry import mapping


class DriverVector(AbstractTabularDriver):
    """
    Driver for vector GIS data.

    Driver keyword arguments (``driver_kwargs``) to the request dataset:

    * ``'feature_class'`` --> For File Geodatabases, a string feature class name is required.
    """
    extensions = ('.*\.shp',)
    key = DriverKey.VECTOR
    output_formats = [constants.OutputFormatName.OCGIS, constants.OutputFormatName.SHAPEFILE]
    common_extension = 'shp'

    def init_variable_value(self, variable):
        value = self.get_variable_value(variable)

        if variable.parent is None:
            variable.set_value(list(value.values())[0])
        else:
            for k, v in list(value.items()):
                variable.parent[k].set_value(v)

        # Conform the units if requested.
        for k in list(value.keys()):
            if variable.parent is None:
                v = variable
            else:
                v = variable.parent[k]

            try:
                conform_units_to = self.rd.metadata['variables'][v.source_name].get('conform_units_to')
            except KeyError:
                # This is okay if the target variable is a geometry variable.
                if isinstance(v, GeometryVariable):
                    conform_units_to = None
                else:
                    raise
            if conform_units_to is not None:
                v.cfunits_conform(conform_units_to)

    def get_crs(self, group_metadata):
        crs = group_metadata['crs']
        if len(crs) == 0:
            ret = None
        else:
            ret = CoordinateReferenceSystem(value=crs)
        return ret

    def create_dimension_map(self, group_metadata):
        ret = {DMK.GEOM: {DMK.VARIABLE: VariableName.GEOMETRY_VARIABLE,
                          DMK.DIMENSION: DimensionName.GEOMETRY_DIMENSION}}
        ret = DimensionMap.from_dict(ret)
        crs = self.get_crs(group_metadata)
        if crs is not None:
            ret.set_crs(crs)
        return ret

    def get_source_metadata_as_json(self):
        raise RequestableFeature

    def create_raw_field(self, **kwargs):
        """See superclass :meth:`ocgis.driver.base.AbstractDriver.create_raw_field`."""

        field = super(DriverVector, self).create_raw_field(**kwargs)
        group_metadata = kwargs.get('group_metadata')
        if group_metadata is None:
            group_metadata = self.rd.metadata
        geom_type = group_metadata['schema']['geometry']

        GeometryVariable(name=VariableName.GEOMETRY_VARIABLE, request_dataset=self.rd, parent=field,
                         geom_type=geom_type, dimensions=DimensionName.GEOMETRY_DIMENSION)

        if group_metadata is not None:
            crs = self.get_crs(group_metadata)
            if crs is not None:
                field.add_variable(crs)

        return field

    def get_variable_metadata(self, variable_object):
        if isinstance(variable_object, GeometryVariable):
            # Geometry variables are located in a different metadata section.
            ret = self.metadata_source[variable_object.name]
        else:
            ret = super(DriverVector, self).get_variable_metadata(variable_object)
        return ret

    def get_variable_value(self, variable, as_geometry_iterator=False):
        # Iteration is always based on source indices.
        iteration_dimension = variable.dimensions[0]
        src_idx = iteration_dimension._src_idx
        if src_idx is None:
            raise ValueError("Iteration dimension must have a source index.")
        else:
            if iteration_dimension._src_idx_type == SourceIndexType.BOUNDS:
                src_idx = slice(*src_idx)

        # For vector formats based on loading via iteration, it makes sense to load all values with a single pass.
        with driver_scope(self, slc=src_idx) as g:
            if as_geometry_iterator:
                return (row['geom'] for row in g)

            ret = {}
            if variable.parent is None:
                ret[variable.name] = np.zeros(variable.shape, dtype=variable.dtype)
                for idx, row in enumerate(g):
                    ret[variable.name][idx] = row['properties'][variable.name]
            else:
                ret = {}
                # Initialize the variable data as zero arrays.
                for v in list(variable.parent.values()):
                    if not isinstance(v, CoordinateReferenceSystem):
                        ret[v.name] = np.ma.array(np.zeros(v.shape, dtype=v.dtype), mask=False)
                # Fill those arrays.
                for idx, row in enumerate(g):
                    for dv in list(variable.parent.values()):
                        if isinstance(dv, (CoordinateReferenceSystem, GeometryVariable)):
                            continue
                        dv = dv.name
                        try:
                            ret[dv][idx] = row['properties'][dv]
                        except TypeError:
                            # Property value may be none. Set the data to masked if this is the case.
                            if row['properties'][dv] is None:
                                ret[dv].mask[idx] = True
                            else:
                                raise
                    try:
                        ret[constants.VariableName.GEOMETRY_VARIABLE][idx] = row['geom']
                    except KeyError:
                        pass

        # Only supply a mask if something is actually masked. Otherwise, remove the mask.
        is_masked = any([v.mask.any() for v in ret.values()])
        if not is_masked:
            for k, v in ret.items():
                ret[k] = v.data

        return ret

    @staticmethod
    def _close_(obj):
        from ocgis.spatial.geom_cabinet import GeomCabinetIterator
        if isinstance(obj, GeomCabinetIterator):
            # Geometry iterators have no close methods.
            pass
        else:
            obj.close()

    def _get_metadata_main_(self):
        with driver_scope(self) as data:
            m = data.sc.get_meta(path=self.rd.uri, driver_kwargs=self.rd.driver_kwargs)
            geom_dimension_name = DimensionName.GEOMETRY_DIMENSION
            m['dimensions'] = {geom_dimension_name: {'size': len(data), 'name': geom_dimension_name}}
            m['variables'] = OrderedDict()

            # Groups are not currently supported in vector formats but metadata expects groups.
            m['groups'] = OrderedDict()

            for p, d in list(m['schema']['properties'].items()):
                d = get_dtype_from_fiona_type(d)
                m['variables'][p] = {'dimensions': (geom_dimension_name,), 'dtype': d, 'name': p,
                                     'attrs': OrderedDict()}

            m[VariableName.GEOMETRY_VARIABLE] = {'dimensions': (geom_dimension_name,),
                                                 'dtype': object,
                                                 'name': geom_dimension_name,
                                                 'attrs': OrderedDict()}
        return m

    def _init_variable_from_source_main_(self, variable_object, variable_metadata):
        if is_auto_dtype(variable_object._dtype):
            variable_object.dtype = variable_metadata['dtype']

        variable_attrs = variable_object._attrs
        for k, v in list(variable_metadata['attrs'].items()):
            if k not in variable_attrs:
                variable_attrs[k] = deepcopy(v)

    @staticmethod
    def _open_(uri, mode='r', **kwargs):
        kwargs = kwargs.copy()
        if mode == 'r':
            from ocgis import GeomCabinetIterator
            # The feature class keyword is driver specific.
            if 'feature_class' in kwargs:
                driver_kwargs = {'feature_class': kwargs.pop('feature_class')}
                kwargs[KeywordArgument.DRIVER_KWARGS] = driver_kwargs
            return GeomCabinetIterator(path=uri, **kwargs)
        elif mode in ('a', 'w'):
            ret = fiona.open(uri, mode=mode, **kwargs)
        else:
            raise ValueError('Mode not supported: "{}"'.format(mode))
        return ret

    @classmethod
    def _write_variable_collection_main_(cls, field, opened_or_path, write_mode, **kwargs):

        from ocgis.collection.field import Field

        if not isinstance(field, Field):
            raise ValueError('Only fields may be written to vector GIS formats.')

        fiona_crs = kwargs.get('crs')
        fiona_schema = kwargs.get('fiona_schema')
        fiona_driver = kwargs.get('fiona_driver', 'ESRI Shapefile')
        iter_kwargs = kwargs.pop('iter_kwargs', {})
        iter_kwargs[KeywordArgument.DRIVER] = cls

        # This finds the geometry variable used in the iterator. Need for the general geometry type that may not be
        # determined using the record iterator.
        geom_variable = field.geom
        if geom_variable is None:
            raise ValueError('A geometry variable is required for writing to vector GIS formats.')

        # Open the output Fiona object using overloaded values or values determined at call-time.
        if not cls.inquire_opened_state(opened_or_path):
            if fiona_crs is None:
                if field.crs is not None:
                    fiona_crs = field.crs.value
            _, archetype_record = next(field.iter(**iter_kwargs))
            archetype_record = format_record_for_fiona(fiona_driver, archetype_record)
            if fiona_schema is None:
                fiona_schema = get_fiona_schema(geom_variable.geom_type, archetype_record)
        else:
            fiona_schema = opened_or_path.schema
            fiona_crs = opened_or_path.crs
            fiona_driver = opened_or_path.driver

        # The Fiona GeoJSON driver does not support update.
        if fiona_driver == 'GeoJSON':
            mode = 'w'
        else:
            mode = 'a'

        # Write the template file.
        if fiona_driver != 'GeoJSON':
            if vm.rank == 0 and write_mode != MPIWriteMode.FILL:
                with driver_scope(cls, opened_or_path=opened_or_path, mode='w', driver=fiona_driver, crs=fiona_crs,
                                  schema=fiona_schema) as _:
                    pass

        # Write data on each rank to the file.
        if write_mode != MPIWriteMode.TEMPLATE:
            for rank_to_write in vm.ranks:
                if vm.rank == rank_to_write:
                    with driver_scope(cls, opened_or_path=opened_or_path, mode=mode, driver=fiona_driver,
                                      crs=fiona_crs, schema=fiona_schema) as sink:
                        itr = field.iter(**iter_kwargs)
                        write_records_to_fiona(sink, itr, fiona_driver)
                vm.barrier()


def format_record_for_fiona(driver, record):
    ret = record
    if driver == 'ESRI Shapefile':
        ret = OrderedDict()
        for k, v in list(record.items()):
            if len(k) > 10:
                k = k[0:10]
            ret[k] = v
    return ret


def get_dtype_from_fiona_type(ftype, data_model=None):
    if ftype.startswith('int'):
        ret = get_dtype('int', data_model=data_model)
    elif ftype.startswith('str'):
        ret = object
    elif ftype.startswith('float'):
        ret = get_dtype('float', data_model=data_model)
    else:
        raise NotImplementedError(ftype)
    return ret


def get_fiona_crs(vc_or_field):
    try:
        # Attempt to pull the coordinate system from the field-like object. If it is a variable collection, look for
        # CRS variables.
        ret = vc_or_field.crs.value
    except AttributeError:
        ret = None
        for v in list(vc_or_field.values()):
            if isinstance(v, CoordinateReferenceSystem):
                ret = v
                break
    return ret


def get_fiona_schema(geom_type, archetype_record):
    ret = {'geometry': geom_type, 'properties': OrderedDict()}

    p = ret['properties']
    for k, v in list(archetype_record.items()):
        p[k] = get_fiona_type_from_pydata(v, string_width=constants.FIONA_STRING_LENGTH)

    return ret


def get_fiona_string_width(arr):
    ret = 0
    for ii in arr.flat:
        ii = str(ii)
        if len(ii) > ret:
            ret = len(ii)
    ret = 'str:{}'.format(ret)
    return ret


def get_fiona_type_from_pydata(pydata, string_width=None):
    m = {datetime.date: 'str',
         datetime.datetime: 'str',
         np.int64: 'int',
         np.float64: 'float',
         np.float32: 'float',
         np.float16: 'float',
         np.int16: 'int',
         np.int32: 'int',
         str: 'str',
         np.dtype('int32'): 'int',
         np.dtype('int64'): 'int',
         np.dtype('float32'): 'float',
         np.dtype('float64'): 'float',
         int: 'int',
         float: 'float',
         object: 'str',
         np.dtype('O'): 'str'}

    # Attempt to add unicode for Python 2.
    try:
        m[unicode] = 'str'
    except NameError:
        pass

    if pydata is None:
        # None types may be problematic for output vector format. Set default type to integer.
        ftype = 'int'
    else:
        dtype = type(pydata)
        try:
            ftype = m[dtype]
        except KeyError:
            # This may be a NumPy string type.
            if str(dtype).startswith('|S'):
                ftype = 'str'
            else:
                raise

    if ftype == 'str':
        if string_width is None:
            string_width = len(pydata)
        ftype = ftype + ':' + str(string_width)

    return ftype


def iter_field_slices_for_records(vc_like, dimension_names, variable_names):
    dimensions = [vc_like.dimensions[d] for d in dimension_names]
    target = vc_like.copy()
    to_pop = []
    for v in list(target.values()):
        if v.name not in variable_names:
            to_pop.append(v.name)
    for tp in to_pop:
        target.pop(tp)

    # Load all values from source.
    target.load()

    iterators = [list(range(len(d))) for d in dimensions]
    for indices in itertools.product(*iterators):
        dslice = {d.name: indices[idx] for idx, d in enumerate(dimensions)}
        yield target[dslice]


def get_geometry_variable(field_like):
    """
    :param field_like: A field or variable collection.
    :return: The geometry variable.
    :rtype: GeometryVariable
    :raises: ValueError
    """

    # Find the geometry variable.
    geom = None
    try:
        # Try to get the geometry assuming it is a field object.
        geom = field_like.geom
    except AttributeError:
        for v in list(field_like.values()):
            if isinstance(v, GeometryVariable):
                geom = v
    if geom is None:
        exc = ValueError('A geometry variable is required.')
        ocgis_lh(exc=exc)

    return geom


def write_records_to_fiona(sink, itr, driver):
    for geom, record in itr:
        record = format_record_for_fiona(driver, record)
        record = {'properties': record, 'geometry': mapping(geom)}
        sink.write(record)
