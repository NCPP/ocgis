#!/bin/bash

CPPFLAGS="-I$PREFIX/include" LDFLAGS="-L$PREFIX/lib" \
./configure --prefix=$PREFIX \
            --with-geos=$PREFIX/bin/geos-config \
            --with-static-proj4=$PREFIX \
            --with-netcdf=$PREFIX \
            --with-python \
            --disable-rpath \
            --without-pam

make -j ${CPU_COUNT}
make install

# Copy data files.
mkdir -p $PREFIX/share/gdal/
cp data/*csv $PREFIX/share/gdal/
cp data/*wkt $PREFIX/share/gdal/
