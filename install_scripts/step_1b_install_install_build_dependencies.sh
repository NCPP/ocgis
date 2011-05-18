#!/bin/bash

# install dependencies for building libraries from source
sudo apt-get install -y wget
sudo apt-get install -y unzip
sudo apt-get install -y gcc
sudo apt-get install -y g++
sudo apt-get install -y swig

SRCDIR=~/src

echo ""
echo "Creating a directory for source files..."
if ! [ -e $SRCDIR ]; then
    mkdir $SRCDIR
    echo "... source file directory has been created."
else
    echo "... source file directory already exists."
fi
echo "... finished creating a directory for source files"
echo ""

