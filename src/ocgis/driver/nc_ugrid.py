from ocgis import Variable
from ocgis.constants import DriverKey, DMK, VariableName, Topology, KeywordArgument
from ocgis.driver.nc import DriverNetcdfCF
from ocgis.exc import GridDeficientError
from ocgis.spatial.grid import GridUnstruct


class DriverNetcdfUGRID(DriverNetcdfCF):
    """
    Driver for NetCDF data following the UGRID convention. It will also interpret CF convention for axes not overloaded
    by UGRID.
    """

    _priority = False
    key = DriverKey.NETCDF_UGRID

    @staticmethod
    def create_host_attribute_variable(dimension_map, name=VariableName.UGRID_HOST_VARIABLE):
        point = dimension_map.get_topology(Topology.POINT)
        dkeys = [DMK.X, DMK.Y, DMK.LEVEL]
        face_coordinates = None
        if point is not None and point != Topology.AUTO:
            x_repr, y_repr, z_repr = [point.get_variable(k) for k in dkeys]
            if x_repr is not None:
                face_coordinates = [x_repr, y_repr]
                if z_repr is not None:
                    face_coordinates.append(z_repr)
                face_coordinates = ' '.join(face_coordinates)

        poly = dimension_map.get_topology(Topology.POLYGON)
        x, y, z = [poly.get_variable(k) for k in dkeys]
        if x is None:
            node_coordinates = None
        else:
            node_coordinates = [x, y]
            if z is not None:
                node_coordinates.append(z)
            node_coordinates = ' '.join(node_coordinates)

        face_node_connectivity = poly.get_variable(DMK.ELEMENT_NODE_CONNECTIVITY)

        if z is None:
            dimension = 2
        else:
            dimension = 3

        locations = []
        if point is not None:
            locations.append('face')
        if poly is not None:
            locations.append('node')
        locations = ' '.join(locations)

        attrs = {'standard_name': 'mesh_topology',
                 'cf_role': 'mesh_topology',
                 'dimension': dimension,
                 'locations': locations,
                 'node_coordinates': node_coordinates,
                 'face_coordinates': face_coordinates,
                 'face_node_connectivity': face_node_connectivity}

        return Variable(name=name, attrs=attrs)

    def create_dimension_map(self, group_metadata, strict=False):
        dmap = super(DriverNetcdfUGRID, self).create_dimension_map(group_metadata, strict=strict)
        variables = group_metadata['variables']

        # Find the attribute host.
        attr_host = None
        for v in variables.values():
            if v['attrs'].get('cf_role') == 'mesh_topology':
                attr_host = v
                dmap.set_variable(DMK.ATTRIBUTE_HOST, v['name'], attrs=v['attrs'].copy())
        if attr_host is None:
            raise ValueError('Attribute host variable not found on UGRID file.')

        # Check for representative coordinates.
        target_host_attr = 'face_coordinates'
        set_coordinate_dmap_variables(attr_host, dmap, target_host_attr, variables, Topology.POINT)

        # Check for nodes.
        target_host_attr = 'node_coordinates'
        has_nodes = set_coordinate_dmap_variables(attr_host, dmap, target_host_attr, variables, Topology.POLYGON)
        if has_nodes:
            face_node_connectivity = attr_host['attrs'].get('face_node_connectivity')
            tdmap = dmap.get_topology(Topology.POLYGON)
            tdmap.set_variable(DMK.ELEMENT_NODE_CONNECTIVITY, face_node_connectivity,
                               dimension=variables[face_node_connectivity]['dimensions'][0])
        return dmap

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata):
        ret = None
        poly = dimension_map.get_topology(Topology.POLYGON, create=False)
        if poly is not None:
            ret = poly.get_variable(DMK.ELEMENT_NODE_CONNECTIVITY)
            if ret is not None:
                ret = poly.get_dimension(DMK.ELEMENT_NODE_CONNECTIVITY)[0]

        if ret is None:
            line = dimension_map.get_topology(Topology.LINE, create=False)
            if line is not None:
                ret = line.get_variable(DMK.ELEMENT_NODE_CONNECTIVITY)
                if ret is not None:
                    ret = line.get_dimension(DMK.ELEMENT_NODE_CONNECTIVITY)[0]

        if ret is None:
            point = dimension_map.get_topology(Topology.POLYGON, create=False)
            if point is not None:
                ret = point.get_dimension(DMK.X)[0]

        if ret is None:
            msg = 'Cannot identify distributed dimension. Checked element, x, and representative x dimensions.'
            raise ValueError(msg)
        return ret

    @staticmethod
    def get_grid(field):
        try:
            ret = GridUnstruct(parent=field)
        except GridDeficientError:
            ret = None
        return ret

    @classmethod
    def _get_field_write_target_(cls, field):
        """Collective!"""

        if field.crs is not None:
            field.crs.format_spatial_object(field)

        attr_host = field.dimension_map.get_variable(DMK.ATTRIBUTE_HOST)
        if attr_host is None or attr_host not in field:
            attr_host = cls.create_host_attribute_variable(field.dimension_map)
            field.dimension_map.set_variable(DMK.ATTRIBUTE_HOST, attr_host)
            field.add_variable(attr_host)

        return field


def set_coordinate_dmap_variables(attr_host, dmap, target_host_attr, variables, topology):
    """
    Set coordinate variables on the target dimension map. This will pass-through if the the target host attribute
    contains no coordinate variable names. Return ``True`` if the dimension map was updated.

    :param dict attr_host: The attribute host variable metadata.
    :param dmap: Dimension map to set variable on.
    :type dmap: :class:`~ocgis.DimensionMap`
    :param str target_host_attr: Name of the host attribute containing coordinate variable names.
    :param dict variables: Group-level metadata for variables.
    :param topology: The destination topology for setting on the dimension map.
    :type topology: :class:`~ocgis.constants.Topology`
    :rtype: bool
    """
    coordinates = attr_host.get('attrs', {}).get(target_host_attr)
    ret = False
    dmap_keys = [DMK.X, DMK.Y, DMK.LEVEL]
    if coordinates is not None:
        coordinates = coordinates.split(' ')
        tdmap = dmap.get_topology(topology, create=True)
        for idx, fc in enumerate(coordinates):
            tdmap.set_variable(dmap_keys[idx], fc, dimension=variables[fc][KeywordArgument.DIMENSIONS][0])
        ret = True
    return ret
