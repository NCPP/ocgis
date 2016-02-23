#!/usr/bin/env bash

docker pull continuumio/anaconda
docker run -it -v ~/l/ocgis:/ocgis continuumio/anaconda bash

cd /ocgis/conda-recipes

# Build using IOOS
conda build -c ioos ocgis

# Build without IOOS
apt-get update && apt-get install build-essential
conda build ocgis

anaconda login
anaconda upload -u nesii -c ocgis `conda build --output gdal fiona ocgis`