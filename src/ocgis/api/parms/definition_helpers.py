class MetadataAttributes(object):
    """
    Manages a mapping of attribute names to values. Unless a the full dictionary is provided (i.e. with "variable" and
    "field" keys), the input dictionary to ``other`` is assumed to target the variable.

    >>> other = {"hello": "world"}
    >>> ma = MetadataAttributes(other=other)
    >>> ma.value
    >>> {"variable": {"hello": "world"}, "field": {}}

    :param dict other: The dictionary object to place into the the managed mapping.
    """
    _keys = ('variable', 'field')
    _key_variable_index = 0

    def __init__(self, other=None):
        self.value = {key: {} for key in self._keys}
        if other is not None:
            self.merge(other)

    def __repr__(self):
        return self.value.__repr__()

    def merge(self, other):
        other = other.copy()
        is_hierarchical = False
        for k, v in other.iteritems():
            if isinstance(v, dict):
                is_hierarchical = True
                break
        if is_hierarchical:
            for k, v in other.iteritems():
                self.value[k] = v
        else:
            self.value[self._keys[self._key_variable_index]] = other
