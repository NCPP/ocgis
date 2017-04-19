import datetime
import itertools
from collections import OrderedDict
from copy import deepcopy

import fiona
import numpy as np
from shapely.geometry import mapping

from ocgis import constants, vm
from ocgis.constants import MPIWriteMode, DimensionName, KeywordArgument, DriverKey
from ocgis.driver.base import driver_scope, AbstractTabularDriver
from ocgis.util.helpers import is_auto_dtype
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import SourcedVariable, VariableCollection
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.geom import GeometryVariable


class DriverVector(AbstractTabularDriver):
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
                conform_units_to = self.rd.metadata['variables'][v.name].get('conform_units_to')
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

    def get_dimension_map(self, group_metadata):
        ret = {'geom': {'variable': DimensionName.GEOMETRY_DIMENSION}}
        crs = self.get_crs(group_metadata)
        if crs is not None:
            ret['crs'] = {'variable': crs.name}
        return ret

    def get_source_metadata_as_json(self):
        # tdk: test on vector and netcdf
        raise NotImplementedError

    def get_variable_collection(self, **kwargs):
        parent = VariableCollection(**kwargs)
        for n, v in list(self.metadata_source['variables'].items()):
            SourcedVariable(name=n, request_dataset=self.rd, parent=parent)
        GeometryVariable(name=DimensionName.GEOMETRY_DIMENSION, request_dataset=self.rd, parent=parent)
        crs = self.get_crs(self.metadata_source)
        if crs is not None:
            parent.add_variable(crs)
        return parent

    def get_variable_metadata(self, variable_object):
        if isinstance(variable_object, GeometryVariable):
            # Geometry variables are located in a different metadata section.
            ret = self.metadata_source[variable_object.name]
        else:
            ret = super(DriverVector, self).get_variable_metadata(variable_object)
        return ret

    def get_variable_value(self, variable):
        # Iteration is always based on source indices. Generate them if they are not available on the variable.
        iteration_dimension = variable.dimensions[0]
        if iteration_dimension._src_idx is None:
            raise ValueError("Iteration dimension must have a source index.")
        else:
            src_idx = iteration_dimension._src_idx

        # For vector formats based on loading via iteration, it makes sense to load all values with a single pass.
        with driver_scope(self, slc=src_idx) as g:
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
                        ret[constants.DimensionName.GEOMETRY_DIMENSION][idx] = row['geom']
                    except KeyError:
                        pass
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
            m = data.sc.get_meta(path=self.rd.uri)
            geom_dimension_name = DimensionName.GEOMETRY_DIMENSION
            m['dimensions'] = {geom_dimension_name: {'size': len(data), 'name': geom_dimension_name}}
            m['variables'] = OrderedDict()

            # Groups are not currently supported in vector formats but metadata expects groups.
            m['groups'] = OrderedDict()

            for p, d in list(m['schema']['properties'].items()):
                d = get_dtype_from_fiona_type(d)
                m['variables'][p] = {'dimensions': (geom_dimension_name,), 'dtype': d, 'name': p,
                                     'attributes': OrderedDict()}

            m[geom_dimension_name] = {'dimensions': (geom_dimension_name,),
                                      'dtype': object,
                                      'name': geom_dimension_name,
                                      'attributes': OrderedDict()}
        return m

    def _init_variable_from_source_main_(self, variable_object, variable_metadata):
        if is_auto_dtype(variable_object._dtype):
            variable_object.dtype = variable_metadata['dtype']

        variable_attrs = variable_object._attrs
        for k, v in list(variable_metadata['attributes'].items()):
            if k not in variable_attrs:
                variable_attrs[k] = deepcopy(v)

    @staticmethod
    def _open_(uri, mode='r', **kwargs):
        if mode == 'r':
            from ocgis import GeomCabinetIterator
            return GeomCabinetIterator(path=uri, **kwargs)
        elif mode in ('a', 'w'):
            ret = fiona.open(uri, mode=mode, **kwargs)
        else:
            raise ValueError('Mode not supported: "{}"'.format(mode))
        return ret

    @classmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, write_mode, **kwargs):

        from ocgis.collection.field import OcgField

        if not isinstance(vc, OcgField):
            raise ValueError('Only fields may be written to vector GIS formats.')

        fiona_crs = kwargs.get('crs')
        fiona_schema = kwargs.get('fiona_schema')
        fiona_driver = kwargs.get('fiona_driver', 'ESRI Shapefile')
        iter_kwargs = kwargs.pop('iter_kwargs', {})
        iter_kwargs[KeywordArgument.DRIVER] = cls

        # This finds the geometry variable used in the iterator. Need for the general geometry type that may not be
        # determined using the record iterator.
        geom_variable = vc.geom
        if geom_variable is None:
            raise ValueError('A geometry variable is required for writing to vector GIS formats.')

        # Open the output Fiona object using overloaded values or values determined at call-time.
        if not cls.inquire_opened_state(opened_or_path):
            if fiona_crs is None:
                if vc.crs is not None:
                    fiona_crs = vc.crs.value
            _, archetype_record = next(vc.iter(**iter_kwargs))
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
                        itr = vc.iter(**iter_kwargs)
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


def get_dtype_from_fiona_type(ftype):
    if ftype.startswith('int'):
        ret = np.int
    elif ftype.startswith('str'):
        ret = object
    elif ftype.startswith('float'):
        ret = np.float
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
