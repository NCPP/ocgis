import netCDF4 as nc

from .helpers import get_features_from_fiona, mesh2_to_fiona, write_netcdf, mesh2_to_fiona_iter


def mesh2_nc_to_fiona(in_nc, out_path, crs=None, driver='ESRI Shapefile'):
    """
    :param str in_nc: Path to the input UGRID netCDF file.
    :param str out_path: Path to the output shapefile.
    :param dict crs: A coordinate system dictionary suitable for passing to ``fiona``.
    :param str driver: The ``fiona`` driver name for the output format.
    :returns: Path to the output shapefile.
    :rtype str:
    """

    ds = nc.Dataset(in_nc, 'r')
    try:
        Mesh2_face_nodes = ds.variables['Mesh2_face_nodes'][:]
        Mesh2_node_x = ds.variables['Mesh2_node_x'][:]
        Mesh2_node_y = ds.variables['Mesh2_node_y'][:]
        mesh2_to_fiona(out_path, Mesh2_face_nodes, Mesh2_node_x, Mesh2_node_y, crs=crs, driver=driver)
    finally:
        ds.close()

    return out_path


def mesh2_nc_to_fiona_iter(in_nc):
    """
    :param str in_nc: The path to the input UGRID netCDF file.
    :returns: Yield ``Fiona``-like record dictionaries.
    :rtype: dict
    """

    ds = nc.Dataset(in_nc, 'r')
    try:
        Mesh2_face_nodes = ds.variables['Mesh2_face_nodes'][:]
        Mesh2_node_x = ds.variables['Mesh2_node_x'][:]
        Mesh2_node_y = ds.variables['Mesh2_node_y'][:]
        for record in mesh2_to_fiona_iter(Mesh2_face_nodes, Mesh2_node_x, Mesh2_node_y):
            yield record
    finally:
        ds.close()


def fiona_to_mesh2_nc(in_fiona_path, out_nc_path, nc_format='NETCDF4', driver='ESRI Shapefile'):
    """
    Create a netCDF UGRID file. If geometry coordinates are not counterclockwise, their orientation will be reversed.

    :param str out_nc_path: Full path to the output netCDF file.
    :param str in_fiona_path: Full path to the input shapefile.
    :param str nc_format: The netCDF file format to write. See http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html.
    :param str driver: The input ``fiona`` driver format.
    :returns: Path to the output netCDF file.
    :rtype: str
    """

    features = get_features_from_fiona(in_fiona_path, driver=driver)
    ds = nc.Dataset(out_nc_path, 'w', format=nc_format)
    try:
        write_netcdf(ds, features)
    finally:
        ds.close()
    return out_nc_path
