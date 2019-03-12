import inspect
from collections import OrderedDict

from ocgis.util.addict import Dict


def create_xclim_defn():
    #tdk: doc
    import xclim
    apidef = Dict()

    def pred(x):
        return inspect.isfunction(x)

    for m in inspect.getmembers(xclim.indices, predicate=pred):
        apidef[m[0]]["func"] = m[1]
        apidef[m[0]]["call_approach"] = "function"

    return apidef


def get_xclim_signature(key, xclim_api):
    #tdk:doc
    sigout = Dict()
    target = xclim_api[key]["func"]
    sig = inspect.signature(target)
    theargs = OrderedDict()
    for p, po in sig.parameters.items():
        if po.default == inspect.Parameter.empty:
            theargs[p] = None
        else:
            sigout.kwargs[p] = po.default
    sigout.args = theargs
    return sigout
