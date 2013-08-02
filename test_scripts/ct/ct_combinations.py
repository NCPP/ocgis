import ocgis
from parameters import AbstractParameter, CounterLimit, ConditionalNotMet
from ocgis.util.helpers import itersubclasses
import itertools
from tempfile import mkdtemp
import shutil
from ocgis.exc import DefinitionValidationError


def get_parameters():
    ret = []
    for sc in itersubclasses(AbstractParameter):
        if not sc.__name__.startswith('Abstract'):
            ret.append(sc)
    return(ret)

def get_parameter_class(name):
    for sc in itersubclasses(AbstractParameter):
        if not sc.__name__.startswith('Abstract'):
            if sc.name == name:
                return(sc)

def iter_combinations(start=0,execute=True):
    iterators = [parm() for parm in get_parameters()]
    klasses = get_parameters()
    for ctr,combo in enumerate(itertools.product(*iterators)):
        if ctr < start:
            continue
        dir_output = mkdtemp()
        try:
            kwargs = {'dir_output':dir_output}
            for dct in combo:
                kwargs.update(dct)
            ## check for any hit counters on parameters
            try:
                for klass in klasses:
                    for counter in klass.counters:
                        counter.check(klass.name,kwargs)
            except CounterLimit:
                continue
            ## search for conditional parameters and update appropriately
            try:
                for key,value in kwargs.iteritems():
                    if isinstance(value,basestring) and value.startswith('_conditional_'):
                        klass = get_parameter_class(key)
                        new_value = klass.get_conditional(kwargs,value)
                        kwargs[key] = new_value
            except ConditionalNotMet:
                continue
            ## execute the operation
            try:
                print(ctr)
                ocgis.env.VERBOSE = True
                ops = ocgis.OcgOperations(**kwargs)
                if execute:
                    print(ops)
                    ret = ops.execute()
            except DefinitionValidationError as e:
                check_exception(kwargs,e)
            except:
                raise
        finally:
            shutil.rmtree(dir_output)
            
def check_exception(kwargs,e):
    reraise = True
    if kwargs['output_format'] == 'nc':
        if kwargs['spatial_operation'] != 'intersects':
            reraise = False
        if kwargs['aggregate']:
            reraise = False
        if kwargs['calc_raw']:
            reraise = False
    if kwargs['calc'] is not None:
        ref = kwargs['calc']
        if ref[0]['func'] in ('duration','freq_duration'):
            if 'year' not in kwargs['calc_grouping']:
                reraise = False
        if kwargs['output_format'] == 'nc':
            if ref[0]['func'] == 'freq_duration':
                reraise = False
        if kwargs['calc_raw'] or kwargs['aggregate']:
            if ref[0]['func'] in ('freq_duration',):
                reraise = False
    if reraise:
        raise(e)
        
        
iter_combinations(start=5696,execute=True)