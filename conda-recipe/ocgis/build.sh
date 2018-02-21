#!/usr/bin/env bash


${PYTHON} setup.py test || exit 1;
${PYTHON} setup.py install || exit 1;
