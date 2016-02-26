#!/usr/bin/env bash

# Script should not be run from command line.
exit 1

docker pull bekozi/nbuild-centos6
docker run -it -v ~/project:/project bekozi/nbuild-centos6 bash

cd /project/ocg/git/ocgis/conda-recipes

# Build without IOOS
#conda build -c nesii/channel/ocgis ocgis
conda build ocgis

anaconda login
#anaconda upload -u nesii -c ocgis `conda build --output gdal fiona ocgis udunits2 munch rtree libspatialindex click-plugins cf_units`
anaconda upload -u nesii -c ocgis `conda build --output ocgis icclim`
