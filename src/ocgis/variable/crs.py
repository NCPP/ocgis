import abc
import itertools
import tempfile
from copy import deepcopy

import numpy as np
import six
from fiona.crs import from_string, to_string
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.multipolygon import MultiPolygon

from ocgis import constants
from ocgis.base import AbstractInterfaceObject, raise_if_empty, AbstractOcgisObject
from ocgis.constants import MPIWriteMode, WrappedState, WrapAction, KeywordArgument, CFName, OcgisUnits, \
    ConversionFactor, DimensionMapKey, DMK, OcgisConvention
from ocgis.environment import osr
from ocgis.exc import ProjectionCoordinateNotFound, ProjectionDoesNotMatch, CRSNotEquivalenError, \
    WrappedStateEvalTargetMissing, CRSDepthNotImplemented
from ocgis.spatial.wrap import GeometryWrapper, CoordinateArrayWrapper
from ocgis.util.helpers import iter_array, get_iter

SpatialReference = osr.SpatialReference


@six.add_metaclass(abc.ABCMeta)
class AbstractCRS(AbstractInterfaceObject):
    """
    Base class for all OCGIS coordinate systems. Intended to allow differentiation between standard PROJ.4 coordinate
    systems and specialized OCGIS-supported coordinate systems.
    
    :param angular_units: Angular units for the coordinate system.
    :type angular_units: :attr:`ocgis.constants.OcgisUnits`
    :param linear_units: Linear units for the coordinate system.
    :type linear_units: :attr:`ocgis.constants.OcgisUnits`
    """
    _cf_attributes = None
    _cf_attributes_to_remove = ('units', 'standard_name')
    _default_towgs84 = None
    _fuzzy_value = None
    _string_max_length_global = None

    def __init__(self, angular_units=OcgisUnits.DEGREES, linear_units=None):
        self._angular_units = angular_units
        self._linear_units = linear_units

    @property
    def angular_units(self):
        return self._angular_units

    @property
    def dist(self):
        """Here for variable compatibility."""
        return False

    @property
    def linear_units(self):
        return self._linear_units

    @property
    def attrs(self):
        return {}

    @property
    def has_initialized_parent(self):
        return False

    @property
    def is_empty(self):
        return False

    @abc.abstractproperty
    def is_geographic(self):
        """
        :return: ``True`` if the coordinate system is considered geographic.
        :rtype: bool
        """

    @abc.abstractproperty
    def is_wrappable(self):
        """        
        :return: ``True`` if the coordinate system may be globally wrapped or unwrapped. A wrappable CRS will use degree
         units on ranges 0 to 360 and -180 to 180.
        :rtype: bool
        """

    @property
    def is_orphaned(self):
        return True

    @property
    def ndim(self):
        return 0

    @property
    def units(self):
        raise NotImplementedError

    def as_variable(self):
        from ocgis.variable.base import Variable
        var = Variable(name=self.name)
        return var

    def convert_to_empty(self):
        pass

    def create_dimension_map(self, group_metadata, strict=False):
        from ocgis import DimensionMap
        return DimensionMap()

    def format_spatial_object(self, spatial_obj, is_transform=False):
        """
        :param spatial_obj: The spatial object to format.
        :type spatial_obj: <varying>
        :param bool is_transform: If ``True``, this is a coordinate system transformation format call. CF attributes
         should be removed. If ``False``, this is for a write and attributes should be left as is or overwritten if
         explicitly defined by the coordinate system.
        """
        # No changes to geometry variables.
        from ocgis.variable.geom import GeometryVariable
        if isinstance(spatial_obj, GeometryVariable):
            return

        # Allows grids to be modified in addition to fields.
        if hasattr(spatial_obj, 'data_variables'):
            for data_variable in spatial_obj.data_variables:
                data_variable.attrs[CFName.GRID_MAPPING] = self.name

        # Add CF attributes to coordinate variables.
        if self._cf_attributes is None and is_transform:
            targets = spatial_obj.coordinate_variables
            for target in targets:
                for attr_key in self._cf_attributes_to_remove:
                    target.attrs.pop(attr_key, None)
        elif self._cf_attributes is not None:

            def _update_attr_(target, name, value, clobber=False):
                attrs = target.attrs
                if name not in attrs or (name in attrs and clobber):
                    attrs[name] = value

            updates = {}
            for c in spatial_obj.coordinate_variables:
                i = spatial_obj.dimension_map.inquire_is_xyz(c)
                if i != DMK.LEVEL and i is not None:
                    updates[c] = self._cf_attributes[i]

            for target, name_values in updates.items():
                if target is not None:
                    for k, v in name_values.items():
                        _update_attr_(target, k, v)

    @classmethod
    def fuzzy_check(cls, value):
        """
        Coordinate system definitions are often changing in interpreters. This function allows coordinate system
        definitions to vary slightly, but still be created appropriately.

        This returns ``True`` if there is a fuzzy match.

        :param dict value: The coordinate system dictionary definition.
        :return: bool
        """
        req = cls._fuzzy_value
        ret = True
        for k, v in req.items():
            if k not in value:
                ret = False
                break
            else:
                if v != value[k]:
                    ret = False
                    break
        if ret:
            if cls._default_towgs84 is not None and 'towgs84' in value:
                ret = value['towgs84'] == cls._default_towgs84
        return ret

    @classmethod
    def get_wrap_action(cls, state_src, state_dst):
        """
        :param int state_src: The wrapped state of the source dataset. (:class:`ocgis.constants.WrappedState`)
        :param int state_dst: The wrapped state of the destination dataset. (:class:`ocgis.constants.WrappedState`)
        :returns: The wrapping action to perform on ``state_src``. (:class:`ocgis.constants.WrapAction`)
        :rtype: int
        :raises: NotImplementedError, ValueError
        """

        possible = [WrappedState.WRAPPED, WrappedState.UNWRAPPED, WrappedState.UNKNOWN]
        has_issue = None
        if state_src not in possible:
            has_issue = 'source'
        if state_dst not in possible:
            has_issue = 'destination'
        if has_issue is not None:
            msg = 'The wrapped state on "{0}" is not recognized.'.format(has_issue)
            raise ValueError(msg)

        # the default action is to do nothing.
        ret = None
        # if the wrapped state of the destination is unknown, then there is no appropriate wrapping action suitable for
        # the source.
        if state_dst == WrappedState.UNKNOWN:
            ret = None
        # if the destination is wrapped and src is unwrapped, then wrap the src.
        elif state_dst == WrappedState.WRAPPED:
            if state_src == WrappedState.UNWRAPPED:
                ret = WrapAction.WRAP
        # if the destination is unwrapped and the src is wrapped, the source needs to be unwrapped.
        elif state_dst == WrappedState.UNWRAPPED:
            if state_src == WrappedState.WRAPPED:
                ret = WrapAction.UNWRAP
        else:
            raise NotImplementedError(state_dst)
        return ret

    def get_wrapped_state(self, target):
        """
        :param target: Return the wrapped state of a field. This function only checks grid centroids and geometry
         exteriors. Bounds/corners on the grid are excluded.
        :type target: :class:`~ocgis.Field`
        """
        # TODO: Wrapped state should operate on the x-coordinate variable vectors or geometries only.
        # TODO: This should be a method on grids and geometry variables.
        from ocgis.collection.field import Field
        from ocgis.spatial.base import AbstractXYZSpatialContainer
        from ocgis import vm

        raise_if_empty(self)

        # If this is not a wrappable coordinate system, wrapped state is undefined.
        if not self.is_wrappable:
            ret = None
        else:
            if isinstance(target, Field):
                grid = target.grid
                if grid is not None:
                    target = grid
                else:
                    target = target.geom

            if target is None:
                raise WrappedStateEvalTargetMissing
            elif target.is_empty:
                ret = None
            elif isinstance(target, AbstractXYZSpatialContainer):
                ret = self._get_wrapped_state_from_array_(target.x.get_value())
            else:
                stops = (WrappedState.WRAPPED, WrappedState.UNWRAPPED)
                ret = WrappedState.UNKNOWN
                geoms = target.get_masked_value().flat
                _is_masked = np.ma.is_masked
                _get_ws = self._get_wrapped_state_from_geometry_
                for geom in geoms:
                    if not _is_masked(geom):
                        flag = _get_ws(geom)
                        if flag in stops:
                            ret = flag
                            break

        rets = vm.gather(ret)
        if vm.rank == 0:
            rets = set(rets)
            if WrappedState.WRAPPED in rets:
                ret = WrappedState.WRAPPED
            elif WrappedState.UNWRAPPED in rets:
                ret = WrappedState.UNWRAPPED
            else:
                ret = list(rets)[0]
        else:
            ret = None
        ret = vm.bcast(ret)

        return ret

    def prepare_geometry_variable(self, subset_geom, rhs_tol=10.0, inplace=True):
        """
        Prepared a geometry variable for subsetting. This method:

        * Appropriately wraps subset geometries for spherical coordinate systems.

        :param subset_geom: The geometry variable to prepare.
        :type subset_geom: :class:`~ocgis.GeometryVariable`
        :param float rhs_tol: The amount, in matching coordinate system units, to buffer the right hand selection
         geometry.
        :param bool inplace: If ``True``, modify the object in-place.
        """
        # TODO: This should work with arbitrary geometries not just bounding boxes.
        assert subset_geom.size == 1
        if self.is_wrappable and subset_geom.is_bbox and subset_geom.wrapped_state == WrappedState.UNWRAPPED:
            if not inplace:
                subset_geom = subset_geom.deepcopy()
            svalue = subset_geom.get_value()
            buffered_bbox = svalue[0].bounds
            if buffered_bbox[0] < 0.:
                bbox_rhs = list(deepcopy(buffered_bbox))
                bbox_rhs[0] = buffered_bbox[0] + 360.
                bbox_rhs[2] = 360. + rhs_tol

                bboxes = [buffered_bbox, bbox_rhs]
            else:
                bboxes = [buffered_bbox]
            bboxes = [box(*b) for b in bboxes]
            if len(bboxes) > 1:
                svalue[0] = MultiPolygon(bboxes)

        return subset_geom

    def set_string_max_length_global(self, value=None):
        """Here for variable compatibility."""

    def to_xarray(self, **kwargs):
        """
        Convert the CRS variable to a :class:`xarray.DataArray`. This *does not* traverse the parent's hierararchy. Use
        the conversion method on the variable's parent to convert all variables in the collection.

        :rtype: :class:`xarray.DataArray`
        """
        from xarray import DataArray

        return DataArray(attrs=self.attrs, name=self.name, data=[])

    def wrap_or_unwrap(self, action, target, force=False):
        from ocgis.variable.geom import GeometryVariable
        from ocgis.spatial.grid import Grid

        if action not in (WrapAction.WRAP, WrapAction.UNWRAP):
            raise ValueError('"action" not recognized: {}'.format(action))

        if target.wrapped_state != action or force:
            if action == WrapAction.WRAP:
                attr = 'wrap'
            else:
                attr = 'unwrap'

            if isinstance(target, GeometryVariable):
                w = GeometryWrapper()
                func = getattr(w, attr)
                target_value = target.get_value()
                for idx, target_geom in iter_array(target_value, use_mask=True, return_value=True,
                                                   mask=target.get_mask()):
                    target_value.__setitem__(idx, func(target_geom))
            elif isinstance(target, Grid):
                ca = CoordinateArrayWrapper()
                func = getattr(ca, attr)
                func(target.x.get_value())
                target.remove_bounds()
                if target.has_allocated_point:
                    getattr(target.get_point(), attr)()
                if target.has_allocated_polygon:
                    getattr(target.get_polygon(), attr)()
            else:
                raise NotImplementedError(target)

            target.wrapped_state = "auto"

    @classmethod
    def _get_wrapped_state_from_array_(cls, arr):
        """
        :param arr: Input n-dimensional array.
        :type arr: :class:`numpy.ndarray`
        :returns: Wrapped state enumeration value from :class:`~ocgis.constants.WrappedState`.
        :rtype: int
        """

        gt_m180 = arr > constants.MERIDIAN_180TH
        lt_pm = arr < 0

        if np.any(gt_m180):
            ret = WrappedState.UNWRAPPED
        elif np.any(lt_pm):
            ret = WrappedState.WRAPPED
        else:
            ret = WrappedState.UNKNOWN

        return ret

    @classmethod
    def _get_wrapped_state_from_geometry_(cls, geom):
        """
        :param geom: The input geometry.
        :type geom: :class:`~shapely.geometry.point.Point`, :class:`~shapely.geometry.point.Polygon`,
         :class:`~shapely.geometry.multipoint.MultiPoint`, :class:`~shapely.geometry.multipolygon.MultiPolygon`
        :returns: A string flag. See class level ``_flag_*`` attributes for values.
        :rtype: str
        :raises: NotImplementedError
        """

        if isinstance(geom, BaseMultipartGeometry):
            itr = geom
        else:
            itr = [geom]

        app = np.array([])
        for element in itr:
            if isinstance(element, Point):
                element_arr = [np.array(element)[0]]
            elif isinstance(element, Polygon):
                element_arr = np.array(element.exterior.coords)[:, 0]
            else:
                raise NotImplementedError(type(element))
            app = np.append(app, element_arr)

        return cls._get_wrapped_state_from_array_(app)

    @staticmethod
    def _place_prime_meridian_array_(arr):
        """
        Replace any 180 degree values with the value of :attribute:`ocgis.constants.MERIDIAN_180TH`.

        :param arr: The target array to modify inplace.
        :type arr: :class:`numpy.array`
        :rtype: boolean :class:`numpy.array`
        """
        from ocgis import constants

        # find the values that are 180
        select = arr == 180
        # replace the values that are 180 with the constant value
        np.place(arr, select, constants.MERIDIAN_180TH)
        # return the mask used for the replacement
        return select


@six.add_metaclass(abc.ABCMeta)
class AbstractProj4CRS(AbstractCRS):
    """
    Base class for coordinate systems that may be transformed using PROJ.4.
    """

    @classmethod
    def load_from_metadata(cls, group_metadata):
        """
        Create a coordinate system object using group metadata.

        :param dict group_metadata: The metdata to interpret. This should be for the current group only.
        :return: :class:`~ocgis.CRS`
        """
        names = []
        vmeta = group_metadata['variables']
        for k, v in vmeta.items():
            vattrs = v.get('attrs', {})
            if vattrs.get(OcgisConvention.Name.OCGIS_ROLE) == OcgisConvention.Value.ROLE_COORDSYS:
                if vattrs.get(OcgisConvention.Name.PROJ4) is not None:
                    names.append(k)

        len_names = len(names)
        if len_names > 1:
            raise ValueError('Multiple coordinate system variables found: {}'.format(names))
        elif len_names == 0:
            raise ProjectionDoesNotMatch
        else:
            v = vmeta[names[0]]
            ret = CRS(name=v['name'], proj4=v['attrs'][OcgisConvention.Name.PROJ4])
        return ret


class CoordinateReferenceSystem(AbstractProj4CRS, AbstractInterfaceObject):
    """
    Defines a coordinate system objects. One of ``value``, ``proj4``, or ``epsg`` is required.

    :param value: A dictionary representation of the coordinate system with PROJ.4 paramters as keys.
    :type value: dict
    :param proj4: A PROJ.4 string.
    :type proj4: str
    :param epsg: An EPSG code.
    :type epsg: int
    :param name: A custom name for the coordinate system.
    :type name: str
    """

    def __init__(self, value=None, proj4=None, epsg=None, name=OcgisConvention.Name.COORDSYS):
        self.name = name
        # Allows operations on data variables to look through an empty dimension list. Alleviates instance checking.
        self.dimensions = tuple()
        self.dimension_names = tuple()
        self.has_bounds = False
        self._epsg = epsg

        # Some basic overloaded for WGS84.
        if epsg == 4326:
            value = WGS84().value

        # Add a special check for init keys in value dictionary.
        if value is not None:
            if 'init' in value and list(value.values())[0].startswith('epsg'):
                epsg = int(list(value.values())[0].split(':')[1])
                value = None

        if value is None:
            if proj4 is not None:
                value = from_string(proj4)
            elif epsg is not None:
                sr = SpatialReference()
                sr.ImportFromEPSG(epsg)
                value = from_string(sr.ExportToProj4())
            else:
                msg = 'A value dictionary, PROJ.4 string, or EPSG code is required.'
                raise ValueError(msg)
        else:
            # Remove unicode to avoid strange issues with proj and fiona.
            for k, v in value.items():
                if isinstance(v, six.string_types):
                    value[k] = str(v)
                else:
                    try:
                        value[k] = v.tolist()
                    # this may be a numpy arr that needs conversion
                    except AttributeError:
                        continue

        sr = SpatialReference()
        sr.ImportFromProj4(to_string(value))
        self.value = from_string(sr.ExportToProj4())

        try:
            assert self.value != {}
        except AssertionError:
            msg = 'Empty CRS: The conversion to PROJ.4 may have failed. The CRS value is: {0}'.format(value)
            raise ValueError(msg)

    def __eq__(self, other):
        try:
            if self.sr.IsSame(other.sr) == 1:
                ret = True
            else:
                # Try a value comparison.
                if self.value == other.value:
                    ret = True
                else:
                    # Try without the "wktext" flag.
                    new_values = [self.value.copy(), other.value.copy()]
                    for n in new_values:
                        n.pop('wktext', None)
                    ret = new_values[0] == new_values[1]
        except AttributeError:
            # likely a nonetype of other object type
            if other is None or not isinstance(other, self.__class__):
                ret = False
            else:
                raise
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    # def __repr__(self):
    #     msg = [str(type(self))]
    #     elements = []
    #     elements.append("name  = {}".format(self.name))
    #     elements.append("value = {}".format(self.value))
    #     elements = [' {}'.format(e) for e in elements]
    #     msg.extend(elements)
    #     return '\n'.join(msg)

    @property
    def dtype(self):
        return None

    @property
    def has_allocated_value(self):
        return False

    @property
    def is_geographic(self):
        return bool(self.sr.IsGeographic())

    @property
    def is_wrappable(self):
        return self.is_geographic

    @property
    def linear_units(self):
        return self.sr.GetLinearUnitsName()

    @property
    def proj4(self):
        if self._epsg == 4326:
            ret = WGS84().proj4
        else:
            ret = self.sr.ExportToProj4()
        return ret

    @property
    def sr(self):
        sr = SpatialReference()
        sr.ImportFromProj4(to_string(self.value))
        return sr

    @property
    def shape(self):
        return tuple([0])

    @property
    def size(self):
        return 0

    def as_variable(self, with_proj4=True):
        ret = super(CoordinateReferenceSystem, self).as_variable()
        if with_proj4:
            ret.attrs['proj4'] = self.proj4
        return ret

    def convert_to_empty(self):
        pass

    def extract(self, *args, **kwargs):
        return self

    def get_mask(self):
        return None

    def load(self, *args, **kwargs):
        """Compatibility with variable."""
        pass

    def write_to_rootgrp(self, rootgrp, with_proj4=True, **kwargs):
        """
        Write the coordinate system to an open netCDF file.

        :param rootgrp: An open netCDF dataset object for writing.
        :type rootgrp: :class:`netCDF4.Dataset`
        :param bool with_proj4: If ``True``, write the PROJ.4 string to the coordinate system variable in an attribute
         called "proj4".
        :returns: The netCDF variable object created to hold the coordinate system metadata.
        :rtype: :class:`netCDF4.Variable`
        """
        variable = rootgrp.createVariable(self.name, 'S1')
        if with_proj4:
            setattr(variable, OcgisConvention.Name.PROJ4, self.proj4)
            setattr(variable, OcgisConvention.Name.OCGIS_ROLE, OcgisConvention.Value.ROLE_COORDSYS)
        return variable

    def write(self, *args, **kwargs):
        write_mode = kwargs.pop('write_mode', MPIWriteMode.NORMAL)

        # Let subclasses determine if proj4 should be written.
        with_proj4 = kwargs.pop('with_proj4', None)

        # Fill operations set values on variables. Coordinate system variables have no inherent values constructed only
        # from attributes.
        if write_mode != MPIWriteMode.FILL:
            if with_proj4 is not None:
                ret = self.write_to_rootgrp(*args, **kwargs)
            else:
                ret = self.write_to_rootgrp(*args)
            return ret


class CRS(CoordinateReferenceSystem):
    """Here for convenience."""


@six.add_metaclass(abc.ABCMeta)
class AbstractSphericalCoordinateReferenceSystem(AbstractOcgisObject):
    _cf_attributes = {DimensionMapKey.X: {CFName.UNITS: 'degrees_east',
                                          CFName.STANDARD_NAME: CFName.StandardName.LONGITUDE},
                      DimensionMapKey.Y: {CFName.UNITS: 'degrees_north',
                                          CFName.STANDARD_NAME: CFName.StandardName.LATITUDE}}
    is_wrappable = True
    is_geographic = True

    def format_spatial_object(self, *args, **kwargs):
        if self.angular_units == OcgisUnits.RADIANS:
            msg = '{} are not acceptable units for field formatting. Must be {}.'
            msg = msg.format(OcgisUnits.RADIANS, OcgisUnits.DEGREES)
            raise ValueError(msg)

        super(AbstractSphericalCoordinateReferenceSystem, self).format_spatial_object(*args, **kwargs)

    def transform_coordinates(self, other_crs, x, y, z, inverse=False):
        """
        Transform coordinate arrays to match ``other_crs``. If ``inverse`` is ``True``, then ``other_crs`` is
        considered the defining coordinate system for the input vectors. Coordinate arrays must have matching 
        dimensions.

        .. note:: If ``z`` is a scalar, the transformed ``z`` value is not returned.

        :param other_crs: :class:`ocgis.variable.crs.AbstractCRS`
        :param x: The x-coordinate array.
        :type x: :class:`numpy.ndarray`
        :param y: The y-coordinate array.
        :type y: :class:`numpy.ndarray`
        :param z: The z-coordinate array.
        :type z: :class:`numpy.ndarray`
        :param bool inverse: If ``True``, the coordinate system of the input variables is ``other_crs`` and the
          transform CRS is the object.
        """

        if not isinstance(other_crs, Cartesian):
            raise CRSNotEquivalenError(self, other_crs)

        if inverse:
            z_out = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            x_out = np.arctan2(y, x)
            y_out = np.arcsin(z / z_out)

            if self.angular_units == OcgisUnits.DEGREES:
                x_out *= ConversionFactor.RAD_TO_DEG
                y_out *= ConversionFactor.RAD_TO_DEG
        else:
            if self.angular_units == OcgisUnits.DEGREES:
                x = x.copy()
                y = y.copy()
                select = x > 180.
                if select.any():
                    raise ValueError('x-coordinate must be wrapped if degrees.')
                x *= ConversionFactor.DEG_TO_RAD
                y *= ConversionFactor.DEG_TO_RAD

            x_out = z * np.cos(y) * np.cos(x)
            y_out = z * np.cos(y) * np.sin(x)
            z_out = z * np.sin(y)

        return x_out, y_out, z_out

    def transform_geometry(self, other_crs, geom, inverse=False):
        from ocgis import GeometryVariable
        assert isinstance(geom, GeometryVariable)
        assert geom.geom_type == 'Polygon'
        assert geom.shape == (1,)

        if not inverse:
            if geom.wrapped_state == WrappedState.UNWRAPPED:
                raise ValueError('Geometry coordinates may not be unwrapped.')

        sgeom = geom.get_value()[0]
        coords = np.array(sgeom.exterior)
        if not sgeom.has_z:
            new_coords = np.ones((coords.shape[0], 3), dtype=coords.dtype)
            new_coords[:, 0:2] = coords
        else:
            new_coords = coords
        x = new_coords[:, 0]
        y = new_coords[:, 1]
        z = new_coords[:, 2]

        x_out, y_out, z_out = self.transform_coordinates(other_crs, x, y, z, inverse=inverse)

        x[:] = x_out
        y[:] = y_out
        z[:] = z_out

        new_geom = Polygon(new_coords)
        geom.get_value()[0] = new_geom

    def transform_grid(self, other_crs, grid, inverse=False):
        """
        Transform a grid's coordinate system.
        
        :param other_crs: See :meth:`ocgis.variable.crs.AbstractSphericalCoordinateReferenceSystem.transform_coordinates`
        :param grid: The grid to transform in-place.
        :type grid: :class:`ocgis.Grid`
        :param inverse: See :meth:`ocgis.variable.crs.AbstractSphericalCoordinateReferenceSystem.transform_coordinates` 
        """

        if not inverse:
            wrapped_state = grid.wrapped_state
            if wrapped_state == WrappedState.UNWRAPPED:
                raise ValueError('x-coordinates may not be wrapped')
        if not grid.has_z:
            raise ValueError('A z-/level-coordinate is required')

        grid.expand()
        x = grid.x.get_value()
        y = grid.y.get_value()
        z = grid.z.get_value()

        x_out, y_out, z_out = self.transform_coordinates(other_crs, x, y, z, inverse=inverse)

        if grid.has_bounds:
            xb = grid.x.bounds.get_value()
            yb = grid.y.bounds.get_value()
            zb = grid.z.bounds.get_value()

            xb_out, yb_out, zb_out = self.transform_coordinates(other_crs, xb, yb, zb, inverse=inverse)

        ################################################################################################################
        # Set values

        grid.x.set_value(x_out)
        grid.y.set_value(y_out)
        grid.z.set_value(z_out)
        if grid.has_bounds:
            grid.x.bounds.set_value(xb_out)
            grid.y.bounds.set_value(yb_out)
            grid.z.bounds.set_value(zb_out)


class Spherical(AbstractSphericalCoordinateReferenceSystem, CoordinateReferenceSystem):
    """
    A spherical model of the Earth's surface with equivalent semi-major and semi-minor axes.

    :param semi_major_axis: The radius of the spherical model. The default value is taken from the PROJ.4 (v4.8.0)
     source code (src/pj_ellps.c).
    :type semi_major_axis: float
    :param units: Either degrees or radians.
    :type units: :class:`ocgis.constants.OcgisUnits`
    """
    _default_towgs84 = '0,0,0,0,0,0,0'
    _fuzzy_value = {'a': 6370997, 'b': 6370997, 'proj': 'longlat'}

    def __init__(self, semi_major_axis=6370997.0, angular_units=OcgisUnits.DEGREES):
        value = {'proj': 'longlat', 'towgs84': self._default_towgs84, 'no_defs': '', 'a': semi_major_axis,
                 'b': semi_major_axis}
        CoordinateReferenceSystem.__init__(self, value=value, name='latitude_longitude')
        AbstractCRS.__init__(self, angular_units=angular_units)
        self.semi_major_axis = semi_major_axis


class Cartesian(AbstractCRS):
    """
    A regular Cartesian coordinate system.
    """
    name = 'cartesian'
    is_geographic = True
    is_wrappable = False

    def as_variable(self):
        ret = super(Cartesian, self).as_variable()
        if self.linear_units is not None:
            ret[CFName.UNITS] = str(self.linear_units)
        return ret

    def transform_geometry(self, other_crs, geom, inverse=False):
        if not isinstance(other_crs, (Spherical, Tripole)):
            raise CRSNotEquivalenError(self, other_crs)
        return other_crs.transform_geometry(self, geom, inverse=True)

    def transform_grid(self, other_crs, grid, inverse=False):
        if not isinstance(other_crs, (Spherical, Tripole)):
            raise CRSNotEquivalenError(self, other_crs)
        return other_crs.transform_grid(self, grid, inverse=True)


class Tripole(AbstractSphericalCoordinateReferenceSystem, AbstractCRS):
    """
    A spherical representation of the Earth's surface having three poles (singularities).
    
    :param spherical: The spherical basis for the tripole coordinate system.
    :type spherical: :class:`ocgis.variable.crs.Spherical`
    """
    name = 'tripole'

    def __init__(self, spherical=None):
        if spherical is None:
            spherical = Spherical()
        self.spherical = spherical
        AbstractCRS.__init__(self, linear_units=spherical.linear_units,
                             angular_units=spherical.angular_units)


class WGS84(CoordinateReferenceSystem):
    """
    A representation of the Earth using the WGS84 datum (i.e. EPSG code 4326).
    """
    _cf_attributes = {DimensionMapKey.X: {CFName.UNITS: 'degrees_east',
                                          CFName.STANDARD_NAME: 'longitude'},
                      DimensionMapKey.Y: {CFName.UNITS: 'degrees_north',
                                          CFName.STANDARD_NAME: 'latitude'}}
    _default_towgs84 = '0,0,0,0,0,0,0'
    _fuzzy_value = {'datum': 'WGS84', 'proj': 'longlat'}

    def __init__(self):
        value = {'no_defs': True, 'datum': 'WGS84', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'}
        try:
            super(WGS84, self).__init__(value=value, name='WGS84_EPSG_4326')
        except:
            value['ellps'] = value['datum']
            super(WGS84, self).__init__(value=value, name='WGS84_EPSG_4326')

    def __eq__(self, other):
        ret = super(WGS84, self).__eq__(other)
        if not ret:
            # Versions handle values differently in GDAL/PROJ.4.
            try:
                ret = other.value == {'proj': 'longlat', 'datum': 'WGS84', 'no_defs': True}
            except AttributeError:
                ret = False
        return ret

    @property
    def proj4(self):
        return '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs '


@six.add_metaclass(abc.ABCMeta)
class CFCoordinateReferenceSystem(CoordinateReferenceSystem):
    _find_projection_coordinates = True

    # Alternative grid mapping names to check. Should be a tuple in subclasses.
    _fuzzy_grid_mapping_names = None

    # Default map parameter values.
    map_parameters_defaults = {}

    def __init__(self, **kwds):
        if 'epsg' in kwds:
            raise ValueError('EPSG codes not enabled when creating CF coordinate systems.')

        self.projection_x_coordinate = kwds.pop('projection_x_coordinate', None)
        self.projection_y_coordinate = kwds.pop('projection_y_coordinate', None)

        # Always provide a default name for the CF-based coordinate systems.
        name = kwds.pop('name', self.grid_mapping_name)

        check_keys = list(kwds.keys())
        for key in list(kwds.keys()):
            check_keys.remove(key)
        if len(check_keys) > 0:
            raise ValueError('The keyword parameter(s) "{0}" was/were not provided.')

        self.map_parameters_values = kwds
        crs = {'proj': self.proj_name}
        for k in list(self.map_parameters.keys()):
            if k in self.iterable_parameters:
                v = getattr(self, self.iterable_parameters[k])(kwds[k])
                crs.update(v)
            else:
                try:
                    crs.update({self.map_parameters[k]: kwds[k]})
                except KeyError:
                    # Attempt to load any default map parameter values.
                    crs.update({self.map_parameters[k]: self.map_parameters_defaults[k]})

        super(CFCoordinateReferenceSystem, self).__init__(value=crs, name=name)

    @abc.abstractproperty
    def grid_mapping_name(self):
        str

    @abc.abstractproperty
    def iterable_parameters(self):
        dict

    @abc.abstractproperty
    def map_parameters(self):
        dict

    @abc.abstractproperty
    def proj_name(self):
        str

    def format_standard_parallel(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()

        ret = {}
        try:
            it = iter(value)
        except TypeError:
            it = [value]
        for ii, v in enumerate(it, start=1):
            ret.update({self.map_parameters['standard_parallel'].format(ii): v})
        return ret

    @classmethod
    def get_fuzzy_names(cls):
        ret = list(get_iter(cls.grid_mapping_name))
        if cls._fuzzy_grid_mapping_names is not None:
            ret += list(get_iter(cls._fuzzy_grid_mapping_names))
        return tuple(ret)

    @classmethod
    def load_from_metadata(cls, var, meta, strict=True):

        def _get_projection_coordinate_(target, meta):
            key = 'projection_{0}_coordinate'.format(target)
            for k, v in list(meta['variables'].items()):
                if 'standard_name' in v['attrs']:
                    if v['attrs']['standard_name'] == key:
                        return k
            raise ProjectionCoordinateNotFound(key)

        r_var = meta['variables'][var]
        try:
            # Look for the grid_mapping attribute on the target variable.
            r_grid_mapping = meta['variables'][r_var['attrs']['grid_mapping']]
        except KeyError:
            # Attempt to match the class's grid mapping name across variables if strictness allows.
            if not strict:
                r_grid_mapping = None
                fuzzy_names = cls.get_fuzzy_names()
                for var_name, var_meta in list(meta['variables'].items()):
                    if var_meta.get('name', var_name) in fuzzy_names:
                        r_grid_mapping = var_meta
                        break
                if r_grid_mapping is None:
                    raise ProjectionDoesNotMatch
            else:
                raise ProjectionDoesNotMatch
        try:
            grid_mapping_name = r_grid_mapping['attrs']['grid_mapping_name']
        except KeyError:
            raise ProjectionDoesNotMatch
        if grid_mapping_name != cls.grid_mapping_name:
            raise ProjectionDoesNotMatch

        # get the projection coordinates if not turned off by class attribute.
        if cls._find_projection_coordinates:
            pc_x, pc_y = [_get_projection_coordinate_(target, meta) for target in ['x', 'y']]
        else:
            pc_x, pc_y = None, None

        kwds = r_grid_mapping['attrs'].copy()
        kwds.pop('grid_mapping_name', None)
        kwds['projection_x_coordinate'] = pc_x
        kwds['projection_y_coordinate'] = pc_y

        # add the correct name to the coordinate system
        kwds['name'] = r_grid_mapping['name']

        cls._load_from_metadata_finalize_(kwds, var, meta)

        return cls(**kwds)

    @classmethod
    def _load_from_metadata_finalize_(cls, kwds, var, meta):
        pass

    def write_to_rootgrp(self, rootgrp, with_proj4=False, **kwargs):
        variable = super(CFCoordinateReferenceSystem, self).write_to_rootgrp(rootgrp, with_proj4=with_proj4)
        variable.grid_mapping_name = self.grid_mapping_name
        for k, v in self.map_parameters_values.items():
            if v is None:
                v = ''
            setattr(variable, k, v)
        return variable


class CFSpherical(Spherical, CFCoordinateReferenceSystem):
    grid_mapping_name = 'latitude_longitude'
    iterable_parameters = None
    map_parameters = None
    proj_name = None

    def __init__(self, *args, **kwargs):
        self.map_parameters_values = {}
        Spherical.__init__(self, *args, **kwargs)

    @classmethod
    def load_from_metadata(cls, var, meta, strict=False, depth=1):
        variables = meta['variables']
        if depth == 1:
            variable_attrs = variables[var].get('attrs', {})
            r_grid_mapping = variable_attrs.get('grid_mapping')
            r_grid_mapping_name = variable_attrs.get('grid_mapping_name')
            if cls.grid_mapping_name in (r_grid_mapping, r_grid_mapping_name):
                return cls()
            else:
                raise ProjectionDoesNotMatch
        elif depth == 2:
            # Check for standard names on variables as spherical is often not defined in the metadata file.
            found = {}
            for k, v in cls._cf_attributes.items():
                sn = v[CFName.STANDARD_NAME]
                found[sn] = []
                for vname, vmeta in variables.items():
                    if vmeta['attrs'].get(CFName.STANDARD_NAME) == sn:
                        found[sn].append(vname)
            # There should be only one variable found for each standard name.
            counts = [len(v) for v in found.values()]
            if any([c > 1 for c in counts]) or any([c == 0 for c in counts]):
                raise ProjectionDoesNotMatch
            else:
                return cls()
        else:
            raise CRSDepthNotImplemented(depth)


class CFAlbersEqualArea(CFCoordinateReferenceSystem):
    grid_mapping_name = 'albers_conical_equal_area'
    iterable_parameters = {'standard_parallel': 'format_standard_parallel'}
    map_parameters = {'standard_parallel': 'lat_{0}',
                      'longitude_of_central_meridian': 'lon_0',
                      'latitude_of_projection_origin': 'lat_0',
                      'false_easting': 'x_0',
                      'false_northing': 'y_0'}
    proj_name = 'aea'


class CFLambertConformal(CFCoordinateReferenceSystem):
    grid_mapping_name = 'lambert_conformal_conic'
    iterable_parameters = {'standard_parallel': 'format_standard_parallel'}
    map_parameters = {'standard_parallel': 'lat_{0}',
                      'longitude_of_central_meridian': 'lon_0',
                      'latitude_of_projection_origin': 'lat_0',
                      'false_easting': 'x_0',
                      'false_northing': 'y_0',
                      'units': 'units'}
    map_parameters_defaults = {'false_easting': 0,
                               'false_northing': 0}
    proj_name = 'lcc'

    @classmethod
    def _load_from_metadata_finalize_(cls, kwds, var, meta):
        kwds['units'] = meta['variables'][kwds['projection_x_coordinate']]['attrs'].get('units')


class CFPolarStereographic(CFCoordinateReferenceSystem):
    grid_mapping_name = 'polar_stereographic'
    map_parameters = {'standard_parallel': 'lat_ts',
                      'latitude_of_projection_origin': 'lat_0',
                      'straight_vertical_longitude_from_pole': 'lon_0',
                      'false_easting': 'x_0',
                      'false_northing': 'y_0',
                      'scale_factor': 'k_0'}
    proj_name = 'stere'
    iterable_parameters = {}

    def __init__(self, *args, **kwds):
        if 'scale_factor' not in kwds:
            kwds['scale_factor'] = 1.0
        super(CFPolarStereographic, self).__init__(*args, **kwds)


class CFNarccapObliqueMercator(CFCoordinateReferenceSystem):
    grid_mapping_name = 'transverse_mercator'
    map_parameters = {'latitude_of_projection_origin': 'lat_0',
                      'longitude_of_central_meridian': 'lonc',
                      'scale_factor_at_central_meridian': 'k_0',
                      'false_easting': 'x_0',
                      'false_northing': 'y_0',
                      'alpha': 'alpha'}
    proj_name = 'omerc'
    iterable_parameters = {}

    def __init__(self, *args, **kwds):
        if 'alpha' not in kwds:
            kwds['alpha'] = 360
        super(CFNarccapObliqueMercator, self).__init__(*args, **kwds)


class CFRotatedPole(CFCoordinateReferenceSystem):
    grid_mapping_name = 'rotated_latitude_longitude'
    iterable_parameters = {}
    map_parameters = {'grid_north_pole_longitude': None, 'grid_north_pole_latitude': None}
    proj_name = 'omerc'
    _find_projection_coordinates = False
    _fuzzy_grid_mapping_names = ('rotated_pole', 'rotated_lat_lon')
    _template = '+proj=ob_tran +o_proj=latlon +o_lon_p={lon_pole} +o_lat_p={lat_pole} +lon_0=180 +ellps={ellps}'

    def __init__(self, *args, **kwds):
        super(CFRotatedPole, self).__init__(*args, **kwds)

        # this is the transformation string used in the proj operation
        self._trans_proj = self._template.format(lon_pole=kwds['grid_north_pole_longitude'],
                                                 lat_pole=kwds['grid_north_pole_latitude'],
                                                 ellps=constants.PROJ4_ROTATED_POLE_ELLPS)

    def update_with_rotated_pole_transformation(self, grid, inverse=False):
        """
        :type grid: :class:`~ocgis.Grid`
        :param bool inverse: If ``True``, this is an inverse transformation.
        :rtype: :class:`~ocgis.Grid`
        """
        assert grid.crs.is_geographic or isinstance(grid.crs, CFRotatedPole)

        grid.remove_bounds()
        grid.expand()

        rlon, rlat = get_lonlat_rotated_pole_transform(grid.x.get_value().flatten(), grid.y.get_value().flatten(),
                                                       self._trans_proj, inverse=inverse)

        grid.x.set_value(rlon.reshape(*grid.shape))
        grid.y.set_value(rlat.reshape(*grid.shape))

    def write_to_rootgrp(self, rootgrp, **kwargs):
        """
        .. note:: See :meth:`~ocgis.interface.base.crs.CoordinateReferenceSystem.write_to_rootgrp`.
        """

        variable = super(CFRotatedPole, self).write_to_rootgrp(rootgrp, **kwargs)
        if kwargs.get(KeywordArgument.WITH_PROJ4, False):
            variable.proj4 = ''
            variable.proj4_transform = self._trans_proj
        return variable


def create_crs(value, **kwargs):
    """
    Create a coordinate system object from a dictionary definition. This will create a spherical coordinate system
    object, for example, if there is a fuzzy match with that coordinate system.

    :param dict value: The coordinate system's dictionary definition.
    :param dict kwargs: Optional keyword arguments to the coordinate system's creation.
    :return: :class:`ocgis.variable.crs.AbstractCRS`
    """
    if value is None:
        ret = None
    else:
        if isinstance(value, AbstractCRS):
            value = value.value
        to_check = np.array([Spherical, WGS84], dtype=object)
        match = np.zeros(len(to_check), dtype=bool)
        for idx, t in enumerate(to_check):
            match[idx] = t.fuzzy_check(value)
        found = to_check[match]
        if found.size > 1:
            raise ValueError('Multiple coordinate systems found: {}'.format(found))
        elif found.size == 0:
            ret = CoordinateReferenceSystem(value=value, **kwargs)
        else:
            ret = found[0]()
    return ret


def get_lonlat_rotated_pole_transform(lon, lat, transform, inverse=False, is_vectorized=False):
    """
    Transform longitude and latitude coordinates to/from their rotated pole representation.

    :param lon: Vector of longitude coordinates.
    :param lat: Vector of latitude coordinates.
    :param str transform: The PROJ.4 transform string.
    :param inverse: If ``True``, coordinates are in spherical longitude/latitude and should be transformed to rotated
     pole.
    :return: A tuple with the first element being transformed longitude and the second element being transformed
     latitude.
    :rtype: tuple
    """

    import csv
    import subprocess

    class ProjDialect(csv.excel):
        lineterminator = '\n'
        delimiter = '\t'

    f = tempfile.NamedTemporaryFile(mode='w')
    try:
        writer = csv.writer(f, dialect=ProjDialect)
        if is_vectorized:
            for lon_idx, lat_idx in itertools.product(*[list(range(lon.shape[0])), list(range(lat.shape[0]))]):
                writer.writerow([lon[lon_idx], lat[lat_idx]])
        else:
            for idx in range(lon.shape[0]):
                writer.writerow([lon[idx], lat[idx]])
        f.flush()
        cmd = transform.split(' ')
        cmd.append(f.name)
        if inverse:
            program = 'invproj'
        else:
            program = 'proj'
        cmd = [program, '-f', '"%.6f"', '-m', '57.2957795130823'] + cmd
        capture = subprocess.check_output(cmd)
    finally:
        f.close()

    capture = capture.decode()
    coords = capture.split('\n')
    new_coords = []

    for ii, coord in enumerate(coords):
        coord = coord.replace('"', '')
        coord = coord.split('\t')
        try:
            coord = list(map(float, coord))
        # likely empty string
        except ValueError:
            if coord[0] == '':
                continue
            else:
                raise
        new_coords.append(coord)

    rlon_rlat = np.array(new_coords)
    rlon = rlon_rlat[:, 0]
    rlat = rlon_rlat[:, 1]

    return rlon, rlat
