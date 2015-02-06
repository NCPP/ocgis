#!/bin/sh

# 1. commit any changes
# 2. git checkout desired doc branch
# 3. cd to doc directory

make clean
make html

TDIR=`mktemp -d`
cp -r ./_build/html/* $TDIR
git checkout gh-pages
cd ..
cp -r $TDIR/* .

## perform any necessary adds, etc.
## git status, git add

git commit -a -m 'doc changes'
git checkout -
git push origin gh-pages


