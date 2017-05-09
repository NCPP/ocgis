==========
Change Log
==========

.. _backwards-compatibility-v1.3:

Version ``2.x`` Backwards Compatibility
---------------------------------------

.. note:: Version ``1.3.x`` will be maintained for bug fixes and dependency upgrades. It is recommended for all users to upgrade to ``v2.x``.

Some changes in ``v2.x`` will break backwards compatbility with ``v1.3.x``. These changes are listed below. If any of these changes affect your workflow, please post a `GitHub Issue <https://github.com/NCPP/ocgis/issues>`_ or contact the `support list <mailto:ocgis_info@list.woc.noaa.gov>`_.

* Changed dimension map format. See :ref:`configuring-a-dimension-map` for the new configuration. Use :meth:`~ocgis.DimensionMap.from_old_style_dimension_map` to convert old-style dimension maps.
* Removed :class:`Inspect` object. Use the :meth:`~ocgis.RequestDataset.inspect` method.
* Removed the :class:`RequestDatasetCollection` object in favor of request dataset or field sequences.
* Removed unique dimension identifers (``TID``, ``LID``, etc.) from tabular outputs. Unique geometry identifiers are maintained for foreign key file relationships.
* Removed ``alias`` parameters and attributes. Aliases are replaced by explicit name parameters (see :ref:`rename_variable <request-dataset>` for example).
* Changed default unique identifier for no geometry from ``1`` to ``None``.
* Changed default coordinate system to :class:`~ocgis.crs.Spherical` from :class:`~ocgis.crs.WGS84`. See :ref:`default-coordinate-system` for guidance on OpenClimateGIS coordinate systems.
* Removed ``headers`` argument from operations. The tabular structure has been streamlined in ``v2.x`` by removing extraneous identifier variables.
* Removed global unique identifier as a default property of all variable objects. Dataset geometry identifers are now unique within a subset operation.
* Removed check for `data` (the coordinate masking is still evaluated for empty subsets) masking following a subset to avoid loading all data from file to retrieve the mask.
* Changed logging output directory to a nested ``logs`` directory inside output directory when ``add_auxiliary_files`` is ``True``.
* Changed masked values in tabular formats to ``None`` from the numeric fill value.
* Removed :meth:`RequestDataset.inspect_as_dict` method.
* Changed :ref:`search_radius_mult key` default to ``None``. Point subsetting will now use the point geometry for intersects operations. Point geometries are no longer buffered by default.
* Removed UGRID conversion. Use `ugrid-tools <https://github.com/NESII/ugrid-tools>`_ to convert to ESMF Unstructured Format.
