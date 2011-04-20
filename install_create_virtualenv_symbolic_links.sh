#----------------------------------------------------
#Setting up symbolic links in the Virtual Environment
#----------------------------------------------------
# within the virtual environment, the virtual environment /bin directory is 
# prepended to the system path;
# for example: /home/terickson/.virtualenvs/openclimategis/bin

# TODO: check if the required environment variables are set
# $PROJ_DIR
# $GEOS_DIR
# $HDF5_DIR
# $NETCDF4_DIR
# $VIRTUALENV_DIR

VIRTUALENV_NAME=openclimategis
VIRTUALENV_DIR=$WORKON_HOME/$VIRTUALENV_NAME

sudo ln -t $VIRTUALENV_DIR/bin/test/ $PROJ_DIR/bin/

for f in `ls $PROJ_DIR/bin`; do echo $f; sudo ln -s $PROJ_DIR/bin/$f $VIRTUALENV_DIR/bin/$f; done
for f in `ls $PROJ_DIR/include`; do echo $f; sudo ln -s $PROJ_DIR/include/$f $VIRTUALENV_DIR/include/$f; done
for f in `ls $PROJ_DIR/lib`; do echo $f; sudo ln -s $PROJ_DIR/lib/$f $VIRTUALENV_DIR/lib/$f; done
for f in `ls $PROJ_DIR/share`; do echo $f; sudo ln -s $PROJ_DIR/share/$f $VIRTUALENV_DIR/share/$f; done

for f in `ls $GEOS_DIR/bin`; do echo $f; sudo ln -s $GEOS_DIR/bin/$f $VIRTUALENV_DIR/bin/$f; done
for f in `ls $GEOS_DIR/include`; do echo $f; sudo ln -s $GEOS_DIR/include/$f $VIRTUALENV_DIR/include/$f; done
for f in `ls $GEOS_DIR/lib`; do echo $f; sudo ln -s $GEOS_DIR/lib/$f $VIRTUALENV_DIR/lib/$f; done

for f in `ls $GDAL_DIR/bin`; do echo $f; sudo ln -s $GDAL_DIR/bin/$f $VIRTUALENV_DIR/bin/$f; done
for f in `ls $GDAL_DIR/include`; do echo $f; sudo ln -s $GDAL_DIR/include/$f $VIRTUALENV_DIR/include/$f; done
for f in `ls $GDAL_DIR/lib`; do echo $f; sudo ln -s $GDAL_DIR/lib/$f $VIRTUALENV_DIR/lib/$f; done
for f in `ls $GDAL_DIR/share`; do echo $f; sudo ln -s $GDAL_DIR/share/$f $VIRTUALENV_DIR/share/$f; done

for f in `ls $HDF5_DIR/bin`; do echo $f; sudo ln -s $HDF5_DIR/bin/$f $VIRTUALENV_DIR/bin/$f; done
for f in `ls $HDF5_DIR/include`; do echo $f; sudo ln -s $HDF5_DIR/include/$f $VIRTUALENV_DIR/include/$f; done
for f in `ls $HDF5_DIR/lib`; do echo $f; sudo ln -s $HDF5_DIR/lib/$f $VIRTUALENV_DIR/lib/$f; done
for f in `ls $HDF5_DIR/share`; do echo $f; sudo ln -s $HDF5_DIR/share/$f $VIRTUALENV_DIR/share/$f; done

for f in `ls $NETCDF4_DIR/bin`; do echo $f; sudo ln -s $NETCDF4_DIR/bin/$f $VIRTUALENV_DIR/bin/$f; done
for f in `ls $NETCDF4_DIR/include`; do echo $f; sudo ln -s $NETCDF4_DIR/include/$f $VIRTUALENV_DIR/include/$f; done
for f in `ls $NETCDF4_DIR/lib`; do echo $f; sudo ln -s $NETCDF4_DIR/lib/$f $VIRTUALENV_DIR/lib/$f; done
for f in `ls $NETCDF4_DIR/share`; do echo $f; sudo ln -s $NETCDF4_DIR/share/$f $VIRTUALENV_DIR/share/$f; done

# link the GDAL osgeo package (installed outsite of the virtual environment)
ln -s -T /usr/lib/python2.6/dist-packages/osgeo $VIRTUALENV_DIR/lib/python2.6/site-packages/osgeo


