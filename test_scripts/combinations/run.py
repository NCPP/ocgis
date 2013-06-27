import parameters as parms
from ocgis.util.helpers import itersubclasses
import itertools
from ocgis import OcgOperations, env
import traceback
import tempfile
import os
import shutil
import argparse
from ocgis.exc import ExtentError
import sys


class BlockedCombination(Exception):
    pass


class CombinationError(Exception):
    
    def __init__(self,inumber,kwds):
        self.inumber = inumber
        self.kwds = kwds
        
    def __str__(self):
        msg = '{0} :: {1}'.format(self.inumber,self.kwds)
        return(msg)


class CombinationRunner(object):
    
    def __init__(self,ops_only=False,target_combo=None,remove_output=True,
                 verbose=False,n_only=False):
        self.ops_only = ops_only
        self.target_combo = target_combo
        self.remove_output = True
        self.verbose = verbose
        self.n_only = n_only
    
    def __iter__(self):
        its = [p().__iter__() for p in self.get_parameters()]
        for ii,values in enumerate(itertools.product(*its)):
            if self.n_only:
                yield(ii)
                continue
            if self.target_combo is not None:
                if self.target_combo > ii:
                    continue
            yield(ii)
            kwds = {}
            for val in values:
                ## check for environmental parameters
                if val.keys()[0].isupper():
                    setattr(env,val.keys()[0],val.values()[0])
                else:
                    kwds.update(val)
            if not self.ops_only:
                kwds.update({'dir_output':tempfile.mkdtemp()})
            try:
                try:
                    ops = OcgOperations(**kwds)
                    try:
                        self.check_blocked(ops)
                    except BlockedCombination:
                        continue
                    if self.verbose: print(ii)
                    if self.ops_only:
                        pass
                    else:
                        ret = ops.execute()
                except Exception as e:
                    raise
                    tb = traceback.format_exc()
                    try:
                        self.check_exception(ii,kwds,e,tb)
                    except:
                        raise
            finally:
                if not self.ops_only and self.remove_output:
                    shutil.rmtree(kwds['dir_output'])
                env.reset()
            
    def check_blocked(self,ops):
        ## do not write the whole datasets without a snippet or a selection geometry
        ## for IO heavy outputs.
        if (ops.geom is None or ops.snippet is False) and ops.output_format in ('csv','csv+','shp'):
            raise(BlockedCombination)
        ## only perform calculation tests on subsetted regions
        if ops.geom is None and ops.calc is not None:
            raise(BlockedCombination)
            
    def check_exception(self,ii,kwds,e,tb):
        reraise = True
        if type(e) == AssertionError:
            ## nc files may not be clipped or aggregated - calculations must
            ## be on raw data.
            if kwds['output_format'] == 'nc':
                if kwds['spatial_operation'] == 'clip' or kwds['aggregate'] is True or kwds['calc_raw'] is True:
                    reraise = False
        elif type(e) == NotImplementedError:
            ## groupings are required for calculations
            if kwds['calc'] is not None and kwds['calc_grouping'] is None:
                reraise = False
            ## snippet is not working for time regions
            if kwds['snippet'] and kwds['dataset'].time_region is not None:
                reraise = False
        elif type(e) == ExtentError:
            ## the counties will not have point data in them -- too small
            if kwds['abstraction'] == 'point' and kwds['geom'].key == 'us_counties':
                if kwds['geom'].spatial.geom.shape[0] == 2:
                    reraise = False
        if reraise:
            print('combo = '+str(ii))
            raise(e)
#            import ipdb;ipdb.set_trace()
#            raise(CombinationError(ii,kwds))
        
    def execute(self):
        for combo in self: pass
    
    def get_parameters(self):
        ret = []
        for sc in itersubclasses(parms.AbstractParameter):
            if sc not in [parms.AbstractBooleanParameter,parms.AbstractEnvironmentalParameter]:
#                if issubclass(sc,parms.AbstractEnvironmentalParameter):
#                    ret_env.append(sc)
#                else:
                ret.append(sc)
        return(ret)
    
    
def main(pargs):
    log_filename = os.path.join(os.path.split(__file__)[0],'combination.log')
    with open(log_filename,'w') as f:
        cr = CombinationRunner(target_combo=pargs.combination,ops_only=pargs.ops_only,n_only=pargs.n)
        for ii,combo in enumerate(cr):
            if not pargs.ops_only or pargs.n:
                f.write('{0}\n'.format(combo))
                f.flush()
    print(ii)
    sys.exit()

## argument parsing ############################################################

parser = argparse.ArgumentParser(description='combinatorial test runner for OCGIS')
parser.add_argument('-c','--combination',type=int,help='target start combination',default=0)
parser.add_argument('-oo','--ops-only',help='just count the operations',default=False,action='store_true')
parser.add_argument('-n',help='just count the parameter combinations',default=False,action='store_true')
parser.set_defaults(func=main)

pargs = parser.parse_args()
pargs.func(pargs)
