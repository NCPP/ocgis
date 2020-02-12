[![Build Status](https://travis-ci.org/NCPP/ocgis.svg?branch=master)](https://travis-ci.org/NCPP/ocgis)

For documentation please visit: https://ocgis.readthedocs.io/en/latest/

For additional project information please visit: http://www.earthsystemcog.org/projects/openclimategis/

For questions or to file a bug report, please create a GitHub issue.

# Overview

OpenClimateGIS (OCGIS) is a Python package designed for geospatial manipulation, subsetting, computation, and translation of spatiotemporal datasets stored in local NetCDF files or files served through THREDDS data servers. OpenClimateGIS has a straightforward, request-based API that is simple to use yet complex enough to perform a variety of computational tasks. The software is built entirely from open source packages.

OpenClimateGIS supports many NetCDF metadata conventions (ESRI Shapefiles / File Geodatabases and CSV formats are also supported):
* Climate & Forecast (CF) Grid
* Unstructured Grid (UGRID)
* SCRIP
* ESMF Unstructured

# GIS Capabilities

* Subsetting (intersects and intersection) of climate datasets by bounding box, Shapely geometries, or shapefiles (city centroid, river reach, a single county or watershed, state boundaries).
* Time and level range subsetting. Also allows for arbitrary label-based slicing.
* Single or multi-dataset requests (concatenation).
* Area-weighted aggregation (spatial averaging) to selection geometries.
* Handles CF-based coordinate systems with full support for coordinate transformations (including the rotated pole coordinate system)
* Geometry wrapping and unwrapping to maintain logically consistent longitudinal domains.
* Polygon, line, and point geometric abstractions.

# Data Conversion

* Access to local NetCDF data or data hosted remotely on a THREDDS (OPeNDAP protocol) data server. Only the piece of data selected by an area-of-interest is transferred from the remote server.
* Stream climate data to multiple formats.
* Extensible converter framework to add custom formats.
* Automatic generation of request metadata.
* Push data to a familiar format to perform analysis or keep the data as NumPy arrays, perform analysis, and dump to a supported format.

# Computation

* Extensible computational framework for arbitrary inclusion of NumPy-based calculations.
* Apply computations to entire data arrays or temporal groups.
* Computed data may be streamed to any supported formats.
