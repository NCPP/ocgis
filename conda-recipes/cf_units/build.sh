#!/bin/bash

SITECFG=cf_units/etc/site.cfg
echo "[System]" > $SITECFG
echo "udunits2_path = $PREFIX/lib/libudunits2.so" >> $SITECFG

$PYTHON setup.py install --single-version-externally-managed  --record record.txt