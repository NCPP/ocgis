#!/usr/bin/env bash


BRANCH=master

sudo bash ~/mount.sh

conda update --yes --all
rm -r ~/anaconda2/envs/ocgis
conda create -y -n ocgis -c nesii ocgis esmpy ipython nose
source activate ocgis
pip install logbook
conda remove -y ocgis

cd ~/src/ocgis
git fetch
git checkout ${BRANCH}
git pull
python setup.py install

cd
export OCGIS_DIR_TEST_DATA=/home/ubuntu/data/ocgis_test_data
export OCGIS_DIR_GEOMCABINET=/home/ubuntu/data/ocgis_test_data/shp
rm .noseids
nosetests -vs --with-id -a '!remote' ~/src/ocgis/src/ocgis/test 2>&1 | tee test_ocgis.out