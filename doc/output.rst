==============
Output Formats
==============

There are multiple output formats in which OCGIS output can be saved on disk: shapefiles (shp), comma separated values (csv), a combination of shapefiles and csv (csv-shp), netCDF (nc) and GeoJSON (geojson). The OcgOperations ``output_format`` parameter determines which format is used, and additional options are available through the ``output_format_options`` (only for netCDF for now).

Shapefiles
==========

Comma Separated Values
======================

Shapefiles + Comma Separated Values
===================================

netCDF
======
`netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ is a binary file format widely used in the climate community. It is self-descriptive is the sense that it carries meta-data describing the file content. netCDF files created by OCGIS carry the original's metadata in addition to updating the file's history. Spatial and temporal subsetting or aggregation operations will modify the original dimensions of the file but retain all relevant data on the spatial coordinates, time and level dimensions. New variable created by OCGIS are named after the ``calc`` operation that defined it.

There are two modes for netCDF output, one handling gridded data and another handling multiple unioned geometries. For example, if ``geom`` is a list of world countries, ``calc`` is the annual average and ``dataset`` stores global gridded daily temperature and ``aggregate`` is set to True, then the annual average temperature for each country would be stored in a variable with dimensions (time, ocgis_geom_union), where the ``ocgis_geom_union`` dimension indexes each individual country. Meta data about the geometries (for example its name), will be stored in variables spanning this spatial dimension. Although the netCDF file does not include each region's geometry (point or polygon), the region's centroid coordinates are stored in the latitude and longitude variables.
Instead of ``ocgis_geom_union``, users can change the name of the discrete geometry dimension by setting the ``output_format_options`` called ``geom_dim``.

GeoJSON
=======



