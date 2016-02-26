#!/usr/bin/env bash

if [ ! -f configure ];
then
   # Make the configure file. Need autoreconf, libtool, libexpat-dev for this.
   autoreconf -i --force
fi

./configure --prefix=$PREFIX
make
make check
make install
