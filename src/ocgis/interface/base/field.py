from contextlib import contextmanager
from copy import copy, deepcopy
from collections import deque, OrderedDict
import itertools
import logging
import numpy as np

from shapely.ops import cascaded_union
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point

from ocgis.interface.base.attributes import Attributes
from ocgis.util.helpers import get_default_or_apply, get_none_or_slice, get_formatted_slice, get_reduced_slice, \
    set_name_attributes, iter_array
from ocgis.interface.base.variable import Variable, VariableCollection
from ocgis import SpatialCollection
from ocgis.util.logging_ocgis import ocgis_lh


class Field(Attributes):
    """
    :param variables: A variable collection containing the values for the field.
    :type variables: :class:`~ocgis.interface.base.variable.VariableCollection`
    :param realization: The realization dimension.
    :type realization: :class:`~ocgis.interface.base.dimension.base.VectorDimension`
    :param temporal: The temporal dimension.
    :type temporal: :class:`~ocgis.interface.base.dimension.temporal.TemporalDimension`
    :param level: The level dimension.
    :type level: :class:`~ocgis.interface.base.dimension.base.VectorDimension`
    :param spatial: The spatial dimension.
    :type spatial: :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension`
    :param dict meta: A dictionary containing additional metadata elements.
    :param int uid: A unique identifier for the field.
    :param str name: A string name for the field.
    :param bool regrid_destination: If ``True``, this field should be used as a regrid destination target.
    :param dict attrs: A dictionary of arbitrary key-value attributes.
    """

    _axis_map = {'realization': 0, 'temporal': 1, 'level': 2}
    _axes = ['R', 'T', 'Z', 'Y', 'X']
    _value_dimension_names = ('realization', 'temporal', 'level', 'row', 'column')
    _variables = None

    def __init__(self, variables=None, realization=None, temporal=None, level=None, spatial=None, meta=None, uid=None,
                 name=None, regrid_destination=False, attrs=None):

        if spatial is None:
            msg = 'At least "spatial" is required.'
            raise ValueError(msg)

        Attributes.__init__(self, attrs=attrs)

        self.realization = realization
        self.temporal = temporal
        self.uid = uid
        self.level = level
        self.spatial = spatial
        self.meta = meta or {}
        self.regrid_destination = regrid_destination
        # holds raw values for aggregated datasets.
        self._raw = None
        # add variables - dimensions are needed first for shape checking
        self.variables = variables
        self._name = name

        # flag used in regridding operations. this should be updated by the driver.
        self._should_regrid = False
        # flag used in regridding to indicate if a coordinate system was assigned by the user in the driver.
        self._has_assigned_coordinate_system = False

        # set default names for the dimensions
        name_mapping = {self.realization: 'realization', self.level: 'level'}
        set_name_attributes(name_mapping)

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, 5)
        ret = copy(self)
        ret.realization = get_none_or_slice(self.realization, slc[0])
        ret.temporal = get_none_or_slice(self.temporal, slc[1])
        ret.level = get_none_or_slice(self.level, slc[2])
        ret.spatial = get_none_or_slice(self.spatial, (slc[3], slc[4]))

        ret.variables = self.variables.get_sliced_variables(slc)

        return ret

    @property
    def crs(self):
        return self.spatial.crs

    @property
    def name(self):
        """
        :returns: The name of the field derived from its variables if not provided.
        :rtype str:
        """

        if self._name is None:
            ret = '_'.join([v.alias for v in self.variables.itervalues()])
        else:
            ret = self._name
        return ret

    @name.setter
    def name(self, value):
        self._name = value
    
    @property
    def shape(self):
        """
        :returns: The shape of the field as a five-element tuple: (realization, time, level, row, column)
        :rtype: tuple
        """

        shape_realization = get_default_or_apply(self.realization, len, 1)
        shape_temporal = get_default_or_apply(self.temporal, len, 1)
        shape_level = get_default_or_apply(self.level, len, 1)
        shape_spatial = get_default_or_apply(self.spatial, lambda x: x.shape, (1, 1))
        ret = (shape_realization, shape_temporal, shape_level, shape_spatial[0], shape_spatial[1])
        return ret
    
    @property
    def shape_as_dict(self):
        """
        :returns: The shape of the field as a dictionary with keys corresponding to axis letter designation defined in
         :attr:`~ocgis.interface.base.field.Field._axes` and value as the shape.
        :rtype dict:
        """

        return dict(zip(self._axes, self.shape))

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        if value is None:
            value = VariableCollection()
        else:
            if isinstance(value, Variable):
                value = VariableCollection(variables=[value])

        if not isinstance(value, VariableCollection):
            raise ValueError('The value must be a Variable or VariableCollection object.')

        self._variables = value
        for v in value.itervalues():
            v._field = self
            if v._value is not None:
                assert v._value.shape == self.shape

    def as_spatial_collection(self):
        """
        :returns: A spatial collection containing the field.
        :rtype: :class:`~ocgis.SpatialCollection`
        """

        coll = SpatialCollection()
        # if there are no vector dimensions, there is no need for a melted representation
        coll.add_field(1, None, self, properties=self.spatial.properties, name=self.name)
        return coll

    def get_between(self, dim, lower, upper):
        pos = self._axis_map[dim]
        ref = getattr(self, dim)
        _, indices = ref.get_between(lower, upper, return_indices=True)
        slc = get_reduced_slice(indices)
        slc_field = [slice(None)] * 5
        slc_field[pos] = slc
        ret = self[slc_field]
        return ret
    
    def get_clip(self, polygon, use_spatial_index=True, select_nearest=False):
        return(self._get_spatial_operation_('get_clip', polygon, use_spatial_index=use_spatial_index,
                                            select_nearest=select_nearest))
    
    def get_intersects(self, polygon, use_spatial_index=True, select_nearest=False):
        return(self._get_spatial_operation_('get_intersects', polygon, use_spatial_index=use_spatial_index,
                                            select_nearest=select_nearest))

    def get_iter(self, add_masked_value=True, value_keys=None):
        """
        :param bool add_masked_value: If ``False``, do not yield masked variable values.
        :param value_keys: A sequence of keys if the variable is a structure array.
        :type value_keys: [str, ...]
        :returns: A dictionary for each value for each variable.
        :rtype: dict
        """

        def _get_dimension_iterator_1d_(target):
            attr = getattr(self, target)
            if attr is None:
                ret = [(0, {})]
            else:
                ret = attr.get_iter()
            return ret

        is_masked = np.ma.is_masked

        # value keys occur when the value array is in fact a structured array with field definitions. this occurs with
        # keyed output functions...
        has_value_keys = False if value_keys is None else True

        r_gid_name = self.spatial.name_uid
        r_name = self.name

        for variable in self.variables.itervalues():
            yld = self._get_variable_iter_yield_(variable)
            yld['name'] = r_name
            ref_value = variable.value
            masked_value = ref_value.fill_value
            iters = map(_get_dimension_iterator_1d_, ['realization', 'temporal', 'level'])
            iters.append(self.spatial.get_geom_iter())
            for [(ridx, rlz), (tidx, t), (lidx, l), (sridx, scidx, geom, gid)] in itertools.product(*iters):
                to_yld = deepcopy(yld)
                ref_idx = ref_value[ridx, tidx, lidx, sridx, scidx]

                # determine if the data is masked
                if is_masked(ref_idx):
                    if add_masked_value:
                        ref_idx = masked_value
                    else:
                        continue

                # realization, time, and level values.
                to_yld.update(rlz)
                to_yld.update(t)
                to_yld.update(l)

                # add geometries to the output
                to_yld['geom'] = geom
                to_yld[r_gid_name] = gid

                # the target value is a structure array, multiple value elements need to be added. these outputs do not
                # a specific value, so it is not added. there may also be multiple elements in the structure which
                # changes how the loop progresses.
                if has_value_keys:
                    for ii in range(ref_idx.shape[0]):
                        for vk in value_keys:
                            try:
                                to_yld[vk] = ref_idx.data[vk][ii]
                            # attempt to access the data directly. masked determination is done above.
                            except ValueError:
                                to_yld[vk] = ref_idx.data[vk][ii]
                        yield (to_yld)
                else:
                    to_yld['value'] = ref_idx
                    yield to_yld
                
    def get_shallow_copy(self):
        return copy(self)
    
    def get_spatially_aggregated(self,new_spatial_uid=None):

        def _get_geometry_union_(value):
            to_union = [geom for geom in value.compressed().flat]
            processed_to_union = deque()
            for geom in to_union:
                if isinstance(geom,MultiPolygon) or isinstance(geom,MultiPoint):
                    for element in geom:
                        processed_to_union.append(element)
                else:
                    processed_to_union.append(geom)
            unioned = cascaded_union(processed_to_union)

            ## convert any unioned points to MultiPoint
            if isinstance(unioned,Point):
                unioned = MultiPoint([unioned])

            ret = np.ma.array([[None]],mask=False,dtype=object)
            ret[0,0] = unioned
            return(ret)

        ret = copy(self)
        ## the spatial dimension needs to be deep copied so the grid may be
        ## dereferenced.
        ret.spatial = deepcopy(self.spatial)
        ## this is the new spatial identifier for the spatial dimension.
        new_spatial_uid = new_spatial_uid or 1
        ## aggregate the geometry containers if possible.
        if ret.spatial.geom.point is not None:
            unioned = _get_geometry_union_(ret.spatial.geom.point.value)
            ret.spatial.geom.point._value = unioned
            ret.spatial.geom.point.uid = new_spatial_uid

        if ret.spatial.geom.polygon is not None:
            unioned = _get_geometry_union_(ret.spatial.geom.polygon.value)
            ret.spatial.geom.polygon._value = _get_geometry_union_(ret.spatial.geom.polygon.value)
            ret.spatial.geom.polygon.uid = new_spatial_uid

        ## update the spatial uid
        ret.spatial.uid = new_spatial_uid
        ## there are no grid objects for aggregated spatial dimensions.
        ret.spatial.grid = None
        ret.spatial._geom_to_grid = False
        ## next the values are aggregated.
        shp = list(ret.shape)
        shp[-2] = 1
        shp[-1] = 1
        itrs = [range(dim) for dim in shp[0:3]]
        weights = self.spatial.weights
        ref_average = np.ma.average

        ## old values for the variables will be stored in the _raw container, but
        ## to avoid reference issues, we need to copy the variables
        new_variables = []
        for variable in ret.variables.itervalues():
            r_value = variable.value
            fill = np.ma.array(np.zeros(shp),mask=False,dtype=variable.value.dtype)
            for idx_r,idx_t,idx_l in itertools.product(*itrs):
                fill[idx_r,idx_t,idx_l] = ref_average(r_value[idx_r,idx_t,idx_l],weights=weights)
            new_variable = copy(variable)
            new_variable._value = fill
            new_variables.append(new_variable)
        ret.variables = VariableCollection(variables=new_variables)

        ## the geometry type of the point dimension is now MultiPoint
        ret.spatial.geom.point.geom_type = 'MultiPoint'

        ## we want to keep a copy of the raw data around for later calculations.
        ret._raw = copy(self)

        return(ret)

    def get_time_region(self, time_region):
        ret = copy(self)
        ret.temporal, indices = self.temporal.get_time_region(time_region, return_indices=True)
        slc = [slice(None), indices, slice(None), slice(None), slice(None)]
        variables = self.variables.get_sliced_variables(slc)
        ret.variables = variables
        return ret

    def iter(self):
        """
        :returns: An ordered dictionary with variable values as keys with geometry information.
        :rtype: :class:`collections.OrderedDict`
        :raises: ValueError
        """

        if any([getattr(self, xx) is not None for xx in ['realization', 'temporal', 'level']]):
            msg = 'Use "iter_melted" for fields having dimensions in addition to space.'
            raise ValueError(msg)

        spatial_name_uid = self.spatial.name_uid
        self_uid = self.uid
        for (sridx, scidx, geom, gid) in self.spatial.get_geom_iter():
            yld = OrderedDict([['geom', geom], ['did', self_uid], [spatial_name_uid, gid]])
            for variable in self.variables.itervalues():
                value = variable.value.data[0, 0, 0, sridx, scidx]
                yld[variable.alias] = value
            yield yld

    def write_to_netcdf_dataset(self, dataset, file_only=False, **kwargs):
        """
        Write the field object to an open netCDF dataset object.

        :param dataset: The open dataset object.
        :type dataset: :class:`netCDF4.Dataset`
        :param bool file_only: If ``True``, we are not filling the value variables. Only the file schema and dimension
         values will be written.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` to pass to ``createVariable``. See
         http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
        :raises: ValueError
        """

        if self.realization is not None:
            msg = 'Fields with a realization dimension may not be written to netCDF.'
            raise ValueError(msg)

        @contextmanager
        def name_scope(target, name, axis):
            previous_name = target.name
            previous_axis = target.axis
            try:
                if target.name is None:
                    target.name = name
                if target.axis is None:
                    target.axis = axis
                yield target
            finally:
                target.name = previous_name
                target.axis = previous_axis

        value_dimensions = []
        try:
            with name_scope(self.temporal, 'time', 'T'):
                self.temporal.write_to_netcdf_dataset(dataset, **kwargs)
                value_dimensions.append(self.temporal.name)
        except AttributeError:
            if self.temporal is not None:
                raise

        try:
            with name_scope(self.level, 'level', 'Z'):
                self.level.write_to_netcdf_dataset(dataset, **kwargs)
                if self.level is not None:
                    value_dimensions.append(self.level.name)
        except AttributeError:
            if self.level is not None:
                raise

        try:
            with name_scope(self.spatial.grid.row, 'yc', 'Y'):
                with name_scope(self.spatial.grid.col, 'xc', 'X'):
                    self.spatial.grid.write_to_netcdf_dataset(dataset, **kwargs)
                    value_dimensions.append(self.spatial.grid.row.name)
                    value_dimensions.append(self.spatial.grid.col.name)
        except AttributeError:
            # write the grid.value directly
            if self.spatial.grid.row is None or self.spatial.grid.col is None:
                self.spatial.grid.write_to_netcdf_dataset(dataset, **kwargs)
                value_dimensions.append('yc')
                value_dimensions.append('xc')
            else:
                raise

        try:
            variable_crs = self.spatial.crs.write_to_rootgrp(dataset)
        except AttributeError:
            if self.spatial.crs is not None:
                raise

        kwargs['dimensions'] = value_dimensions
        for variable in self.variables.itervalues():
            kwargs['fill_value'] = variable.fill_value
            nc_variable = dataset.createVariable(variable.alias, variable.dtype, **kwargs)
            if not file_only:
                nc_variable[:] = variable.value
            variable.write_attributes_to_netcdf_object(nc_variable)

            try:
                nc_variable.grid_mapping = variable_crs._name
            except UnboundLocalError:
                if self.spatial.crs is not None:
                    raise

            try:
                nc_variable.units = variable.units
            except TypeError:
                # likely none for the units
                if variable.units is None:
                    nc_variable.units = ''
                else:
                    raise

        self.write_attributes_to_netcdf_object(dataset)

    def _get_spatial_operation_(self, attr, polygon, use_spatial_index=True, select_nearest=False):
        ref = getattr(self.spatial, attr)
        ret = copy(self)
        ret.spatial, slc = ref(polygon, return_indices=True, use_spatial_index=use_spatial_index,
                               select_nearest=select_nearest)
        slc = [slice(None), slice(None), slice(None)] + list(slc)
        ret.variables = self.variables.get_sliced_variables(slc)

        ## we need to update the value mask with the geometry mask
        self._set_new_value_mask_(ret, ret.spatial.get_mask())

        return(ret)

    def _get_value_from_source_(self, *args, **kwargs):
        raise NotImplementedError

    def _get_variable_iter_yield_(self, variable):
        """
        Retrieve variable-level information. Overloaded by derived fields.

        :param variable: The variable containing attributes to extract.
        :type variable: :class:`~ocgis.Variable`
        :returns: A dictionary containing variable field values mapped to keys.
        :rtype: dict
        """

        yld = {'did': self.uid, 'variable': variable.name, 'alias': variable.alias, 'vid': variable.uid}
        return yld
                    
    @staticmethod
    def _set_new_value_mask_(field,mask):
        ret_shp = field.shape
        rng_realization = range(ret_shp[0])
        rng_temporal = range(ret_shp[1])
        rng_level = range(ret_shp[2])
        ref_logical_or = np.logical_or

        for var in field.variables.itervalues():
            if var._value is not None:
                v = var._value
                for idx_r,idx_t,idx_l in itertools.product(rng_realization,rng_temporal,rng_level):
                    ref = v[idx_r,idx_t,idx_l]
                    ref.mask = ref_logical_or(ref.mask,mask)


class DerivedField(Field):
    
    def _get_variable_iter_yield_(self, variable):
        yld = {'cid': variable.uid, 'calc_key': variable.name, 'calc_alias': variable.alias}

        raw_variable = variable.parents.values()[0]
        yld['did'] = self.uid
        yld['variable'] = raw_variable.name
        yld['alias'] = raw_variable.alias
        yld['vid'] = raw_variable.uid
        return yld


class DerivedMultivariateField(Field):
    
    def _get_variable_iter_yield_(self,variable):
        yld = {}
        yld['cid'] = variable.uid
        yld['calc_key'] = variable.name
        yld['calc_alias'] = variable.alias
        yld['did'] = None
        return(yld)
