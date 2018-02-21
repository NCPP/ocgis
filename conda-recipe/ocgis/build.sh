#!/usr/bin/env bash


${PYTHON} setup.py test || exit 1;
${PYTHON} setup.py install --single-version-externally-managed --record=record.txt || exit 1;
