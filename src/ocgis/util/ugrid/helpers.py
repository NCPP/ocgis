from collections import deque

import fiona
import numpy as np
from shapely.geometry import shape, Polygon, mapping
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.point import Point
from shapely.geometry.polygon import orient

from constants import PYUGRID_LINK_ATTRIBUTE_NAME, PYUGRID_CONVENTIONS_VERSION


def convert_multipart_to_singlepart(path_in, path_out, uid=None, new_uid_name=PYUGRID_LINK_ATTRIBUTE_NAME):
    """
    Convert a vector GIS file from multipart to singlepart geometries. The function copies all attributes and
    maintains things like coordinate systems.

    :param str path_in: Path to the input file containing multipart geometries.
    :param str path_out: Path to the output file.
    :param str uid: If provided, use this attribute as the integer unique identifier.
    :param str new_uid_name: Use this name as the default for the create unique identifier if ``uid`` is ``None``.
    """

    with fiona.open(path_in) as source:
        if uid is None:
            source.meta['schema']['properties'][new_uid_name] = 'int'
        with fiona.open(path_out, mode='w', **source.meta) as sink:
            for record in source:
                if uid is None:
                    uid_value = record['id']
                else:
                    uid_value = record['properties'][uid]
                geom = shape(record['geometry'])
                if uid is None:
                    record['properties'][new_uid_name] = uid_value
                if isinstance(geom, BaseMultipartGeometry):
                    for element in geom:
                        record['geometry'] = mapping(element)
                        sink.write(record)
                else:
                    sink.write(record)


def get_update_feature(fid, feature):
    # todo: doc

    # create the geometry object
    obj = shape(feature['geometry'])
    # only polygons are acceptable geometry types
    try:
        assert not isinstance(obj, BaseMultipartGeometry)
    except AssertionError:
        msg = 'Only singlepart geometries allowed. Perhaps "ugrid.convert_multipart_to_singlepart" would be useful?'
        raise ValueError(msg)
    # if the coordinates are not counterclockwise, reverse the orientation
    if not obj.exterior.is_ccw:
        obj = orient(obj)
        feature['geometry'] = mapping(obj)
    # add this to feature dictionary
    feature['geometry']['object'] = obj
    # add a custom feature identifier
    feature['fid'] = fid
    return feature


def iter_edge_nodes(idx_nodes):
    # todo: doc

    for ii in range(len(idx_nodes)):
        try:
            yld = (idx_nodes[ii], idx_nodes[ii + 1])
        # the last node pair requires linking back to the first node
        except IndexError:
            yld = (idx_nodes[-1], idx_nodes[0])
        yield yld


def get_features_from_fiona(in_path, driver='ESRI Shapefile'):
    """
    :param str in_path: Full path to a the target shapefile to extract features from.
    :param str driver: The ``fiona`` driver to use.
    :returns: A ``deque`` containing feature dictionaries. This is the standard record collection as returned by
     ``fiona`` with the addition of a ``fid`` key and ``object`` key in the ``geometry`` dictionary.
    :rtype: :class:`collections.deque`
    """

    # load features from disk constructing geometry objects
    features = deque()
    idx_fid = 0
    with fiona.open(in_path, 'r', driver=driver) as source:
        for feature in source:
            features.append(get_update_feature(idx_fid, feature))
            idx_fid += 1
    return features


def get_mesh2_variables(features):
    """
    :param features: A features sequence as returned from :func:`ugrid.helpers.get_features_from_fiona`.
    :type features: :class:`collections.deque`
    :returns: A tuple of arrays with index locations corresponding to:

    ===== ================ =============================
    Index Name             Type
    ===== ================ =============================
    0     Mesh2_face_nodes :class:`numpy.ma.MaskedArray`
    1     Mesh2_face_edges :class:`numpy.ma.MaskedArray`
    2     Mesh2_edge_nodes :class:`numpy.ndarray`
    3     Mesh2_node_x     :class:`numpy.ndarray`
    4     Mesh2_node_y     :class:`numpy.ndarray`
    5     Mesh2_face_links :class:`numpy.ndarray`
    ===== ================ =============================

    Information on individual variables may be found here: https://github.com/ugrid-conventions/ugrid-conventions/blob/9b6540405b940f0a9299af9dfb5e7c04b5074bf7/ugrid-conventions.md#2d-flexible-mesh-mixed-triangles-quadrilaterals-etc-topology

    :rtype: tuple (see table for array types)
    """

    # construct the links between faces (e.g. neighbors)
    Mesh2_face_links = deque()
    for idx_source in range(len(features)):
        ref_object = features[idx_source]['geometry']['object']
        for idx_target in range(len(features)):
            # skip if it is checking against itself
            if idx_source == idx_target:
                continue
            else:
                # if the objects only touch they are neighbors and share nodes
                if ref_object.touches(features[idx_target]['geometry']['object']):
                    Mesh2_face_links.append([features[idx_source]['fid'], features[idx_target]['fid']])
                else:
                    continue
    # convert to numpy array for faster comparisons
    Mesh2_face_links = np.array(Mesh2_face_links, dtype=np.int32)

    # the number of faces
    nMesh2_face = len(features)
    # for polygon geometries the first coordinate is repeated at the end of the sequence.
    nMaxMesh2_face_nodes = max([len(feature['geometry']['coordinates'][0]) - 1 for feature in features])
    Mesh2_face_nodes = np.ma.array(np.zeros((nMesh2_face, nMaxMesh2_face_nodes), dtype=np.int32), mask=True)
    # the edge mapping has the same shape as the node mapping
    Mesh2_face_edges = np.zeros_like(Mesh2_face_nodes)

    # holds the start and nodes for each edge
    Mesh2_edge_nodes = deque()
    # holds the raw coordinates of the nodes
    Mesh2_node_x = deque()
    Mesh2_node_y = deque()

    # flag to indicate if this is the first face encountered
    first = True
    # global point index counter
    idx_point = 0
    # global edge index counter
    idx_edge = 0
    # holds point geometry objects
    points_obj = deque()
    # loop through each polygon
    for feature in features:
        # reference the face index
        fid = feature['fid']
        # just load everything if this is the first polygon
        if first:
            # store the point values. remember to ignore the last coordinate.
            for ii in range(len(feature['geometry']['coordinates'][0]) - 1):
                coords = feature['geometry']['coordinates'][0][ii]
                # create and store the point geometry object
                points_obj.append(Point(coords[0], coords[1]))
                # store the x and y coordinates
                Mesh2_node_x.append(coords[0])
                Mesh2_node_y.append(coords[1])
                # increment the point index
                idx_point += 1
            # map the node indices for the face
            Mesh2_face_nodes[fid, 0:idx_point] = range(0, idx_point)
            # construct the edges. compress the node slice to remove any masked values at the tail.
            for start_node_idx, end_node_idx in iter_edge_nodes(Mesh2_face_nodes[fid, :].compressed()):
                Mesh2_edge_nodes.append((start_node_idx, end_node_idx))
                idx_edge += 1
            # map the edges to faces
            Mesh2_face_edges[fid, 0:idx_edge] = range(0, idx_edge)
            # switch the loop flag to indicate the first face has been dealt with
            first = False
        else:
            # holds new node coordinates for the face
            new_Mesh2_face_nodes = deque()
            # only search neighboring faces
            neighbor_face_indices = Mesh2_face_links[Mesh2_face_links[:, 0] == fid, 1]
            for ii in range(len(feature['geometry']['coordinates'][0]) - 1):
                # logic flag to indicate if the point has been found
                found = False
                coords = feature['geometry']['coordinates'][0][ii]
                pt = Point(coords[0], coords[1])
                # search the neighboring faces for matching nodes
                for neighbor_face_index in neighbor_face_indices.flat:
                    # break out of loop if the point has been found
                    if found:
                        break
                    # search over the neighboring face's nodes
                    for neighbor_face_node_index in Mesh2_face_nodes[neighbor_face_index, :].compressed():
                        if pt.almost_equals(points_obj[neighbor_face_node_index]):
                            new_Mesh2_face_nodes.append(neighbor_face_node_index)
                            # point is found, no need to continue with loop
                            found = True
                            break
                # add the new node if it has not been found
                if not found:
                    # add the point object to the collection
                    points_obj.append(pt)
                    # add the coordinates of the new point
                    Mesh2_node_x.append(coords[0])
                    Mesh2_node_y.append(coords[1])
                    # append the index of this new point
                    new_Mesh2_face_nodes.append(idx_point)
                    # increment the point index
                    idx_point += 1
            # map the node indices for the face
            Mesh2_face_nodes[fid, 0:len(new_Mesh2_face_nodes)] = new_Mesh2_face_nodes
            # find and map the edges
            new_Mesh2_face_edges = deque()
            for start_node_idx, end_node_idx in iter_edge_nodes(Mesh2_face_nodes[fid, :].compressed()):
                # flag to indicate if edge has been found
                found_edge = False
                # search existing edge-node combinations accounting for ordering
                for idx_edge_nodes, edge_nodes in enumerate(Mesh2_edge_nodes):
                    # swap the node ordering
                    if edge_nodes == (start_node_idx, end_node_idx) or edge_nodes == (end_node_idx, start_node_idx):
                        new_Mesh2_face_edges.append(idx_edge_nodes)
                        found_edge = True
                        break
                if not found_edge:
                    Mesh2_edge_nodes.append((start_node_idx, end_node_idx))
                    new_Mesh2_face_edges.append(idx_edge)
                    idx_edge += 1
            # update the face-edge mapping
            Mesh2_face_edges[fid, 0:len(new_Mesh2_face_edges)] = new_Mesh2_face_edges

    return Mesh2_face_nodes, \
           Mesh2_face_edges, \
           np.array(Mesh2_edge_nodes, dtype=np.int32), \
           np.array(Mesh2_node_x, dtype=np.float32), \
           np.array(Mesh2_node_y, dtype=np.float32), \
           np.array(Mesh2_face_links, dtype=np.int32)


def mesh2_to_fiona(out_path, Mesh2_face_nodes, Mesh2_node_x, Mesh2_node_y, crs=None, driver='ESRI Shapefile'):
    # todo: doc

    schema = {'geometry': 'Polygon', 'properties': {}}
    with fiona.open(out_path, 'w', driver=driver, crs=crs, schema=schema) as f:
        for feature in mesh2_to_fiona_iter(Mesh2_face_nodes, Mesh2_node_x, Mesh2_node_y):
            f.write(feature)
    return out_path


def mesh2_to_fiona_iter(Mesh2_face_nodes, Mesh2_node_x, Mesh2_node_y):
    # todo: doc

    for feature_idx in range(Mesh2_face_nodes.shape[0]):
        coordinates = deque()
        for node_idx in Mesh2_face_nodes[feature_idx, :].compressed():
            coordinates.append((Mesh2_node_x[node_idx], Mesh2_node_y[node_idx]))
        polygon = Polygon(coordinates)
        feature = {'id': feature_idx, 'properties': {}, 'geometry': mapping(polygon)}
        yield feature


def write_netcdf(ds, features):
    """
    Write to an open dataset object.

    :param ds: :class:`netCDF4.Dataset`
    :param features: A feature sequence as returned from :func:`ugrid.helpers.get_features_from_fiona`.
    """

    start_index = 0

    Mesh2_face_nodes, Mesh2_face_edges, Mesh2_edge_nodes, Mesh2_node_x, \
    Mesh2_node_y, Mesh2_face_links = get_mesh2_variables(features)
    nMesh2_node = ds.createDimension('nMesh2_node', size=Mesh2_node_x.shape[0])
    nMesh2_edge = ds.createDimension('nMesh2_edge', size=Mesh2_edge_nodes.shape[0])
    nMesh2_face = ds.createDimension('nMesh2_face', size=Mesh2_face_nodes.shape[0])
    nMesh2_face_links = ds.createDimension('nMesh2_face_links', size=Mesh2_face_links.shape[0])
    nMaxMesh2_face_nodes = ds.createDimension('nMaxMesh2_face_nodes', size=Mesh2_face_nodes.shape[1])
    Two = ds.createDimension('Two', size=2)
    vMesh2 = ds.createVariable('Mesh2', np.int32)
    vMesh2.cf_role = "mesh_topology"
    vMesh2.long_name = "Topology data of 2D unstructured mesh"
    vMesh2.topology_dimension = 2
    vMesh2.node_coordinates = "Mesh2_node_x Mesh2_node_y"
    vMesh2.face_node_connectivity = "Mesh2_face_nodes"
    vMesh2.edge_node_connectivity = "Mesh2_edge_nodes"
    vMesh2.edge_coordinates = "Mesh2_edge_x Mesh2_edge_y"
    # vMesh2.face_coordinates = "Mesh2_face_x Mesh2_face_y"
    # vMesh2.face_edge_connectivity = "Mesh2_face_edges"
    # vMesh2.face_face_connectivity = "Mesh2_face_links"
    vMesh2_face_nodes = ds.createVariable('Mesh2_face_nodes',
                                          Mesh2_face_nodes.dtype,
                                          dimensions=(nMesh2_face._name, nMaxMesh2_face_nodes._name),
                                          fill_value=Mesh2_face_nodes.fill_value)
    vMesh2_face_nodes[:] = Mesh2_face_nodes
    vMesh2_face_nodes.cf_role = "face_node_connectivity"
    vMesh2_face_nodes.long_name = "Maps every face to its corner nodes."
    vMesh2_face_nodes.start_index = start_index
    vMesh2_edge_nodes = ds.createVariable('Mesh2_edge_nodes', Mesh2_edge_nodes.dtype,
                                          dimensions=(nMesh2_edge._name, Two._name))
    vMesh2_edge_nodes[:] = Mesh2_edge_nodes
    vMesh2_edge_nodes.cf_role = "edge_node_connectivity"
    vMesh2_edge_nodes.long_name = "Maps every edge to the two nodes that it connects."
    vMesh2_edge_nodes.start_index = start_index
    vMesh2_face_edges = ds.createVariable('Mesh2_face_edges', Mesh2_face_edges.dtype,
                                          dimensions=(nMesh2_face._name, nMaxMesh2_face_nodes._name),
                                          fill_value=Mesh2_face_edges.fill_value)
    vMesh2_face_edges[:] = Mesh2_face_edges
    vMesh2_face_edges.cf_role = "face_edge_connectivity"
    vMesh2_face_edges.long_name = "Maps every face to its edges."
    vMesh2_face_edges.start_index = start_index
    vMesh2_face_links = ds.createVariable('Mesh2_face_links', Mesh2_face_links.dtype,
                                          dimensions=(nMesh2_face_links._name, Two._name))
    vMesh2_face_links[:] = Mesh2_face_links
    vMesh2_face_links.cf_role = "face_face_connectivity"
    vMesh2_face_links.long_name = "Indicates which faces are neighbors."
    vMesh2_face_links.start_index = start_index
    vMesh2_node_x = ds.createVariable('Mesh2_node_x', Mesh2_node_x.dtype,
                                      dimensions=(nMesh2_node._name,))
    vMesh2_node_x[:] = Mesh2_node_x
    vMesh2_node_x.standard_name = "longitude"
    vMesh2_node_x.long_name = "Longitude of 2D mesh nodes."
    vMesh2_node_x.units = "degrees_east"
    vMesh2_node_y = ds.createVariable('Mesh2_node_y', Mesh2_node_y.dtype,
                                      dimensions=(nMesh2_node._name,))
    vMesh2_node_y[:] = Mesh2_node_y
    vMesh2_node_y.standard_name = "latitude"
    vMesh2_node_y.long_name = "Latitude of 2D mesh nodes."
    vMesh2_node_y.units = "degrees_north"

    # add global variables
    ds.Conventions = PYUGRID_CONVENTIONS_VERSION
