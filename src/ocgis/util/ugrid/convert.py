import netCDF4 as nc

from helpers import get_features_from_shapefile, mesh2_to_shapefile, write_to_netcdf_dataset


def mesh2_nc_to_shapefile(in_nc, out_path):
    """
    :param str in_nc: Path to the input UGRID netCDF file.
    :param str out_path: Path to the output shapefile.
    :returns: Path to the output shapefile.
    :rtype str:
    """

    ds = nc.Dataset(in_nc, 'r')
    try:
        Mesh2_face_nodes = ds.variables['Mesh2_face_nodes'][:]
        Mesh2_node_x = ds.variables['Mesh2_node_x'][:]
        Mesh2_node_y = ds.variables['Mesh2_node_y'][:]
        mesh2_to_shapefile(out_path, Mesh2_face_nodes, Mesh2_node_x, Mesh2_node_y)
    finally:
        ds.close()

    return out_path


def shapefile_to_mesh2_nc(out_nc_path, shp_path, frmt='NETCDF4'):
    """
    :param str out_nc_path: Full path to the output netCDF file.
    :param str shp_path: Full path to the input shapefile.
    :param str frmt: The netCDF file format to write. See http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html.
    :returns: Path to the output netCDF file.
    :rtype: str
    """

    features = get_features_from_shapefile(shp_path)
    ds = nc.Dataset(out_nc_path, 'w', format=frmt)
    try:
        write_to_netcdf_dataset(ds, features)
    finally:
        ds.close()
    return out_nc_path
