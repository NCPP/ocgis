# Current Release Notes #

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

# Previous Release Notes #

...