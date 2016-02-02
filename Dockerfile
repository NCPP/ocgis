FROM continuumio/miniconda

MAINTAINER ben.koziol@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install build-essential \
                       gfortran

RUN conda install -c nesii/channel/icclim -c nesii/channel/dev-ocgis -c ioos ocgis icclim esmpy nose ipython
RUN pip install ipdb

RUN conda remove ocgis
RUN git clone -b next https://github.com/NCPP/ocgis.git
RUN cd ocgis && python setup.py install
RUN cd .. && rm -r ocgis

ENV GDAL_DATA /opt/conda/share/gdal

RUN python -c "from ocgis.test import run_all; run_all(verbose=False)"

RUN rm -r /opt/conda/pkgs/*
