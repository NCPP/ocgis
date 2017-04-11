from rtree import index
from shapely.prepared import prep


class SpatialIndex(object):
    """
    Create and access spatial indexes using the :mod:`rtree` module.
    """

    def __init__(self, path=None):
        if path is None:
            self._index = index.Index()
        else:
            self._index = index.Rtree(path)

    def add(self, id_geom, shapely_geom):
        """
        ..note: Both parameters may come in as sequences of the appropriate type.

        :param int id_geom: The unique identifier for the input geometry.
        :param :class:`shapely.geometry.Geometry` shapely_geom: The geometry to add to the spatial index. The bounds
         attribute of the geometry is added to the index.
        """
        try:
            self._index.insert(id_geom, shapely_geom.bounds)
        except AttributeError:
            # likely a sequence
            _insert = self._index.insert
            for ig, sg in zip(id_geom, shapely_geom):
                _insert(ig, sg.bounds)

    def iter_intersects(self, shapely_geom, arr, keep_touches=True):
        """
        Return an interator for the unique identifiers of the geometries intersecting the target geometry.

        :param :class:`shapely.geometry.Geometry` shapely_geom: The geometry to use for subsetting. It is the ``bounds``
         attribute fo the geometry that is actually tested.
        :param dict geom_mapping: The collection of geometries to do the full intersects test on. The keys of the
         dictionary correspond to the integer unique identifiers. The values are Shapely geometries.
        :param bool keep_touches: If ``True``, return the unique identifiers of geometries only touching the subset
         geometry.
        :returns: Generator yield integer unique identifiers.
        :rtype: int
        """
        # tdk: update doc
        # Create the geometry iterator. If it is a multi-geometry, we want to iterator over those individually.
        try:
            itr = iter(shapely_geom)
        except TypeError:
            # likely not a multi-geometry
            itr = [shapely_geom]

        for shapely_geom_sub in itr:
            # Return the initial identifiers that intersect with the bounding box using the "rtree" internal method.
            indices = self._get_intersection_rtree_(shapely_geom_sub)
            # Prepare the geometry for faster operations.
            prepared = prep(shapely_geom_sub)
            r_intersects = prepared.intersects
            r_touches = shapely_geom_sub.touches
            for idx in indices:
                geom = arr[idx]
                if r_intersects(geom):
                    if not keep_touches:
                        if not r_touches(geom):
                            yield idx
                    else:
                        yield idx

    def iter_rtree_intersection(self, shapely_geom):
        # Create the geometry iterator. If it is a multi-geometry, we want to iterator over those individually.
        try:
            itr = iter(shapely_geom)
        except TypeError:
            # likely not a multi-geometry
            itr = [shapely_geom]

        for shapely_geom_sub in itr:
            # Return the initial identifiers that intersect with the bounding box using the retree internal method.
            ids = self._get_intersection_rtree_(shapely_geom_sub)
            for idd in ids:
                yield idd

    def _get_intersection_rtree_(self, shapely_geom):
        return self._index.intersection(shapely_geom.bounds)
