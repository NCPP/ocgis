from urlparse import parse_qs

from ocgis import OcgOperations
from ocgis.api.parms.base import AbstractParameter
from ocgis.api.parms.definition import Dataset
from ocgis.util.helpers import itersubclasses


class QueryInterface(object):
    """
    Provides conversion between URL query strings and OpenClimateGIS operations.

    :param qs: The URL query string.
    :type qs: str

    >>> qs = 'uri=data.nc&spatial_operation=intersects'
    """

    _expected_missing_keys = ('uri', 'variable', 'alias')

    def __init__(self, qs):
        self.query_string = qs
        self.query_dict = parse_qs(qs)

    def get_operations(self):
        """
        :returns: An operations objects created by parsing the query string.
        :rtype: :class:`ocgis.api.operations.OcgOperations`
        """

        pmap = {klass.name: klass for klass in itersubclasses(AbstractParameter)}
        kwds = {}

        kwds[Dataset.name] = Dataset.from_query(self)

        for k, v in self.query_dict.iteritems():
            try:
                kwds[k] = pmap[k].from_query(self)
            except KeyError:
                # Some parameters require arguments different from the parameter name.
                if k not in self._expected_missing_keys:
                    raise
        ops = OcgOperations(**kwds)
        return ops
