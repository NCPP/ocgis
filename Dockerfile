FROM continuumio/miniconda

MAINTAINER ben.koziol@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install build-essential \
                       gfortran

RUN conda install -c nesii/channel/icclim -c nesii/channel/dev-ocgis -c ioos ocgis icclim esmpy nose ipython krb5
RUN pip install ipdb

RUN conda remove ocgis
RUN git clone -b master-dev --depth=10 https://github.com/NCPP/ocgis.git
RUN cd ocgis && python setup.py install

ENV GDAL_DATA /opt/conda/share/gdal

RUN cd && nosetests -a '!slow,!remote,!data' /ocgis/src/ocgis/test

RUN rm -r /opt/conda/pkgs/*
