# Current Release Notes #

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

# Previous Release Notes #

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