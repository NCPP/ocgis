FROM continuumio/miniconda3

MAINTAINER ben.koziol@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get clean

RUN conda update -y --all

RUN conda create -y -n ocgis -c conda-forge ocgis esmpy nose mock
RUN bash -c "source activate ocgis && python -c 'import ocgis'"
RUN bash -c "source activate ocgis && conda remove ocgis"
RUN git clone --depth=1 https://github.com/NCPP/ocgis.git
WORKDIR ocgis
RUN bash -c "source activate ocgis && python setup.py install"

ENV GDAL_DATA /opt/conda/share/gdal
RUN bash -c "source activate ocgis && python -c 'from ocgis.test import run_simple; run_simple(verbose=False)'"
RUN bash -c "source activate ocgis && mpirun -n 2 python -c 'from ocgis.test import run_mpi_nodata; run_mpi_nodata(verbose=True)'"
