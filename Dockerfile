FROM continuumio/miniconda

MAINTAINER ben.koziol@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
#RUN apt-get -y install build-essential \
#                       gfortran
RUN apt-get clean

RUN conda update -y --all
RUN conda install -y -c nesii -c conda-forge ocgis esmpy=7.0.0 icclim=4.2.5 nose

RUN conda remove -y ocgis
RUN git clone --depth=1 https://github.com/NCPP/ocgis.git
RUN cd ocgis && python setup.py install

ENV GDAL_DATA /opt/conda/share/gdal
RUN cd && python -c "from ocgis.test import run_simple; run_simple(verbose=False)"

RUN rm -r /opt/conda/pkgs/*
RUN rm -r /ocgis