# Release Notes and Known Bugs #

* **Quicklinks**
 * [v0.06b](#v006b)
 * [v0.05.1b](#v0051b)
 * [v0.05b](#v005b)

## v0.06b ##

### What's New ###
* SQL access to shapefile selection for lower read delays on large shapefiles
  - Shapefiles now require a unique, integer UGID field.
  - ShpProcess object added to assist in shapefile conversion (requires Fiona).
* Header dump of netCDF file attributes included in disk outputs (*_source_metdata.txt).
* Additional attributes in *_did.csv.
* Added "headers" parameters to OcgOperations to allow selection of file headers.
* Added "time_region" to RequestDataset to subset by arbitrary month/year combinations.

### Known Bugs ###
* Missing documentation for ShpProcess
* Outdated documentation for ShpCabinet
* Resolution of NcGridMatrixDimensions fails

## v0.05.1b ##

### What's New ###
* New backend interface to support additional output formats.
* Keyed output removed in favor or "csv+" format.
* Prototype support for large array computations.
* Improved verbose output.
* Support for additional projections.
* Multiple projections supported in a single request.
* Addition of input dataset descriptor CSV file in output data package.
* Multi-file netCDF support using netCDF4.MFDataset.
* Support for point-based subsetting.

# Previous Release Notes #

## v0.05b ##

### What's New ###
1. New implementations for parameter definitions with improved type checking and formatting.
2. Improved metadata formatting.
3. Numerous bug fixes.
4. Added "dir_output" parameter to OcgOperations.
5. Selection geometries are now a derived class of list and carry projection information.
6. Projection information is now written in CF formatting to NetCDF outputs.
7. Computations may be written to NetCDF.
8. OcgOperations parameter "prefix" now names the output folder as well as the files contained in the output directory.
9. Projection support is available again with selection geometries projected to match input dataset.
10. Simple multiprocessing implementation available with all objects serializable.
11. OcgOperations parameter "select_ugid" applied at geometry load-time.
12. New ocgis.env with support of system environment variables.
13. Setup.py has option for autmated install on Ubuntu Linux systems.
14. URL representation of OcgOperations included in metadata output.

## v0.04b ##

### What's New ###
1. RequestDataset and RequestDatasetCollection objects to replace old dataset dictionary in OcgOperations.
   * Time and level ranges subset arguments merged into RequestDataset.
2. Documentation on GitHub: http://ncpp.github.com/ocgis/
3. A "setup.py" module and installation instructions: http://ncpp.github.com/ocgis/install.html
4. Improvements to ShpCabinet adding ability to load all or no attributes.
5. Alpha implementation of URL-to-OcgOperations (and vice versa) capability.
6. Changes to naming conventions of "env" variables.
7. Usage examples in documentation pages: http://ncpp.github.com/ocgis/examples.html.

### Additional Notes ###
1. NetCDF output for calculations not working.
2. Code optimization is behind the multi-dataset implementation.
4. Datasets with projections currently require customizations.
5. "setup.py" uninstall not working.

## v0.03a ##

### What's New ###
1. Added ability to request datasets having differing dimensional domains.
2. Requested dataset may also have differing time and level ranges.
3. Improved 'keyed' output accounting for potentially differing dimensional information.
4. Added additional attributes to melted iterator.
5. Simplified interface that uses requested variable dimenionsal information and attributes to find bounds.
 * Removed unnecessary interface parameters to account for new approach.
6. NetCDF output now copies attributes correctly from originating dataset.

### Additional Notes ###
1. NetCDF output for calculations not working.
2. Multivariate calculations currently broken.
3. Code optimization is behind the multi-dataset implementation.
4. Datasets with projections currently require customizations.

...
