#!/bin/bash

# 1. commit any changes
# 2. git checkout desired doc branch
# 3. cd to doc directory

make clean
make html

TDIR=`mktemp -d`
cp -r ./_build/html/* $TDIR
git checkout gh-pages

git push
git checkout -

