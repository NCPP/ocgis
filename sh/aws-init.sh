#!/usr/bin/env bash

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install tree gfortran git wget

sudo mount /dev/xvdf ~/data
echo "sudo mount /dev/xvdf ~/data" > ~/mount.sh

mkdir -p ~/src
cd ~/src
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
conda update --yes --all
