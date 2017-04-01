import itertools
from collections import namedtuple
from copy import deepcopy


def itr_row(key, sequence):
    for element in sequence:
        yield ({key: element})


def itr_products_keywords(keywords, as_namedtuple=False):
    if as_namedtuple:
        yld_tuple = namedtuple('ITesterKeywords', list(keywords.keys()))

    iterators = [itr_row(ki, vi) for ki, vi in keywords.items()]
    for dictionaries in itertools.product(*iterators):
        yld = {}
        for dictionary in dictionaries:
            yld.update(dictionary)
        if as_namedtuple:
            yld = yld_tuple(**yld)
        yield deepcopy(yld)
