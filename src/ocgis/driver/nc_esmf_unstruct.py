import numpy as np

from ocgis import vm
from ocgis.constants import DriverKey, CFName, OcgisConvention, Topology, DMK, MPITag
from ocgis.driver.base import AbstractUnstructuredDriver
from ocgis.driver.nc import AbstractDriverNetcdfCF
from ocgis.variable.base import Variable, VariableCollection
from ocgis.variable.dimension import create_distributed_dimension, Dimension


class DriverESMFUnstruct(AbstractUnstructuredDriver, AbstractDriverNetcdfCF):
    """
    Driver for the NetCDF-based ESMF Unstructured Format: http://www.earthsystemmodeling.org/esmf_releases/last_built/ESMF_refdoc/node3.html#SECTION03028200000000000000.
    """
    _priority = False
    key = DriverKey.NETCDF_ESMF_UNSTRUCT

    def create_dimension_map(self, group_metadata, strict=False):
        dmap = super(AbstractDriverNetcdfCF, self).create_dimension_map(group_metadata, strict=strict)
        topo = dmap.get_topology(Topology.POLYGON, create=True)

        name_element_conn = 'elementConn'
        varmeta = group_metadata['variables']
        topo.set_variable(DMK.ELEMENT_NODE_CONNECTIVITY, name_element_conn,
                          dimension=varmeta[name_element_conn]['dimensions'][0])
        vattrs = topo.get_attrs(DMK.ELEMENT_NODE_CONNECTIVITY)
        vattrs.pop(CFName.STANDARD_NAME, None)
        vattrs[OcgisConvention.Name.ELEMENT_NODE_COUNT] = 'numElementConn'

        # The ESMF unstructured format stored coordinates together in a variable. This requires sections.
        node_coords_name = 'nodeCoords'
        node_coords_dimension = varmeta[node_coords_name]['dimensions'][0]
        topo.set_variable(DMK.X, node_coords_name, dimension=[node_coords_dimension], section=(None, 0))
        topo.set_variable(DMK.Y, node_coords_name, dimension=[node_coords_dimension], section=(None, 1))

        return dmap

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata):
        # TODO: Consider options for distributing ESMF Unstructured Format.
        return None

    @staticmethod
    def get_element_dimension(gc):
        return gc.parent.dimensions['elementCount']

    @staticmethod
    def get_multi_break_value(cindex):
        """See :meth:`ocgis.spatial.geomc.AbstractGeometryCoordinates.multi_break_value`"""
        return cindex.attrs.get('polygon_break_value')

    @classmethod
    def _get_field_write_target_(cls, field):
        """
        Takes field data out of the OCGIS unstructured format (similar to UGRID) converting to the format expected
        by ESMF Unstructured metadata.
        """

        # The driver for the current field must be NetCDF UGRID to ensure interpretability.
        assert field.dimension_map.get_driver() == DriverKey.NETCDF_UGRID
        grid = field.grid
        # Three-dimensional data is not supported.
        assert not grid.has_z
        # Number of coordinate dimension. This will be 3 for three-dimensional data.
        coord_dim = Dimension('coordDim', 2)

        # Transform ragged array to one-dimensional array. #############################################################

        cindex = grid.cindex
        elements = cindex.get_value()
        num_element_conn_data = [e.shape[0] for e in elements.flat]
        length_connection_count = sum(num_element_conn_data)
        esmf_element_conn = np.zeros(length_connection_count, dtype=elements[0].dtype)
        start = 0

        tag_start_index = MPITag.START_INDEX

        # Collapse the ragged element index array into a single dimensioned vector. This communication block finds the
        # size for the new array. ######################################################################################

        if vm.size > 1:
            max_index = max([ii.max() for ii in elements.flat])
            if vm.rank == 0:
                vm.comm.isend(max_index + 1, dest=1, tag=tag_start_index)
                adjust = 0
            else:
                adjust = vm.comm.irecv(source=vm.rank - 1, tag=tag_start_index)
                adjust = adjust.wait()
                if vm.rank != vm.size - 1:
                    vm.comm.isend(max_index + 1 + adjust, dest=vm.rank + 1, tag=tag_start_index)

        # Fill the new vector for the element connectivity. ############################################################

        for ii in elements.flat:
            if vm.size > 1:
                if grid.archetype.has_multi:
                    mbv = cindex.attrs[OcgisConvention.Name.MULTI_BREAK_VALUE]
                    replace_breaks = np.where(ii == mbv)[0]
                else:
                    replace_breaks = []
                ii = ii + adjust
                if len(replace_breaks) > 0:
                    ii[replace_breaks] = mbv

            esmf_element_conn[start: start + ii.shape[0]] = ii
            start += ii.shape[0]

        # Create the new data representation. ##########################################################################

        connection_count = create_distributed_dimension(esmf_element_conn.size, name='connectionCount')
        esmf_element_conn_var = Variable(name='elementConn', value=esmf_element_conn, dimensions=connection_count)
        esmf_element_conn_var.attrs[CFName.LONG_NAME] = 'Node indices that define the element connectivity.'
        mbv = cindex.attrs.get(OcgisConvention.Name.MULTI_BREAK_VALUE)
        if mbv is not None:
            esmf_element_conn_var.attrs['polygon_break_value'] = mbv
        esmf_element_conn_var.attrs['start_index'] = grid.start_index
        ret = VariableCollection(variables=field.copy().values(), force=True)

        # Rename the element count dimension.
        original_name = ret[cindex.name].dimensions[0].name
        ret.rename_dimension(original_name, 'elementCount')

        # Add the element-node connectivity variable to the collection.
        ret.add_variable(esmf_element_conn_var)

        num_element_conn = Variable(name='numElementConn',
                                    value=num_element_conn_data,
                                    dimensions=cindex.dimensions[0],
                                    attrs={CFName.LONG_NAME: 'Number of nodes per element.'})
        ret.add_variable(num_element_conn)

        node_coords = Variable(name='nodeCoords', dimensions=(grid.node_dim, coord_dim))
        node_coords.units = 'degrees'
        node_coords.attrs[CFName.LONG_NAME] = 'Node coordinate values indexed by element connectivity.'
        node_coords.attrs['coordinates'] = 'x y'
        fill = node_coords.get_value()
        fill[:, 0] = grid.x.get_value()
        fill[:, 1] = grid.y.get_value()
        ret.pop(grid.x.name)
        ret.pop(grid.y.name)
        ret.add_variable(node_coords)

        ret.attrs['gridType'] = 'unstructured'
        ret.attrs['version'] = '0.9'

        return ret
