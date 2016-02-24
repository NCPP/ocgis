#!/usr/bin/env bash


pushd /home/ubuntu/project/ocg/git/ocgis/conda-recipes

#name=bekozi/nbuild-centos6
#file=./Dockerfile-CentOS6-Builder
#docker build -t ${name} --file ${file} .
#docker push ${name}

name=bekozi/ntest-ubuntu
file=./Dockerfile-Ubuntu-Tester
docker build -t ${name} --file ${file} .
docker push ${name}

popd
