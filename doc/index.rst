.. OpenClimateGIS documentation master file, created by
   sphinx-quickstart on Tue Jan 22 09:05:11 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

OpenClimateGIS is a Python package designed for geoprocessing and computation on CF_-compliant climate datasets. It is "request-based" in its current release meaning an operation is entirely defined prior to execution. This approach allows the software to have nearly equivalent functionality from its Python and RESTful APIs.

Currently, there is a single point of entry to OpenClimateGIS: the :class:`~ocgis.OcgOperations` object. This documentation will describe the Python syntax as well as the equivalent RESTful form of its argumentation.

There is additional project content for OpenClimateGIS hosted on it's `CoG Site`_. Please contact ben.koziol@noaa.gov for information, help, or other general inquiries.

.. _CF: http://cf-pcmdi.llnl.gov/
.. _CoG Site: http://www.earthsystemcog.org/projects/openclimategis/

Contents:

.. toctree::
   :maxdepth: 2

   contact
   links
   install
   api
   computation
   examples

.. Indices and tables
..  ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

