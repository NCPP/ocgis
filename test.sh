#!/usr/bin/env bash

#export OCGIS_DIR_TEST_DATA=/home/ubuntu/data/ocgis_test_data
#export OCGIS_DIR_GEOMCABINET=/home/ubuntu/data/ocgis_test_data/shp

rm .noseids
nosetests -vs --with-id -a '!remote' src/ocgis/test 2>&1 | tee /tmp/test_ocgis.out
