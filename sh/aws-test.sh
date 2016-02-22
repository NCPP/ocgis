#!/usr/bin/env bash


conda update --yes --all
conda create -n ocgis -c nesii/channel/dev-ocgis -c nesii/channel/icclim -c ioos ocgis icclim esmpy==7.0.0 ipython nose
source activate ocgis
pip install logbook
sudo bash ~/mount.sh
conda remove -y ocgis

cd ~/src/ocgis
git fetch
git checkout next
git pull
python setup.py install

cd
export OCGIS_DIR_TEST_DATA=/home/ubuntu/data/ocgis_test_data
export OCGIS_DIR_GEOMCABINET=/home/ubuntu/data/ocgis_test_data/shp
rm .noseids
nosetests -vs --with-id ~/src/ocgis/src/ocgis/test 2>&1 | tee test_ocgis.out