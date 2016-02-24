FROM continuumio/miniconda

MAINTAINER ben.koziol@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install build-essential \
                       gfortran
RUN apt-get clean

RUN conda update -y --all
RUN conda install -y -c nesii/channel/ocgis -c nesii ocgis esmpy nose

RUN conda remove -y ocgis
RUN git clone -b master --depth=1 https://github.com/NCPP/ocgis.git
RUN cd ocgis && python setup.py install

ENV GDAL_DATA /opt/conda/share/gdal
RUN cd && nosetests -a '!slow,!remote,!data' /ocgis/src/ocgis/test

RUN rm -r /opt/conda/pkgs/*
RUN rm -r /ocgis