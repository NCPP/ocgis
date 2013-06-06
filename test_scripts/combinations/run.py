import parameters as parms
from ocgis.util.helpers import itersubclasses
import itertools
from ocgis import OcgOperations, env
import traceback
import tempfile
import os
import shutil
import argparse


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
                 verbose=True):
        self.ops_only = ops_only
        self.target_combo = target_combo
        self.remove_output = True
        self.verbose = verbose
    
    def __iter__(self):
        its = [p().__iter__() for p in self.get_parameters()]
        for ii,values in enumerate(itertools.product(*its)):
            if self.target_combo is not None:
                if self.target_combo > ii:
                    continue
            kwds = {}
            for val in values: kwds.update(val)
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
                        yld = (ii,ops)
                    else:
                        ret = ops.execute()
                        yld = (ii,ops,ret)
                    yield(yld)
                except Exception as e:
                    tb = traceback.format_exc()
                    try:
                        self.check_exception(ii,kwds,e,tb)
                    except:
                        raise
            finally:
                if not self.ops_only and self.remove_output:
                    shutil.rmtree(kwds['dir_output'])
            
    def check_blocked(self,ops):
        ## do not write the whole datasets without a snippet or a selection geometry
        if (ops.geom is None or ops.snippet is False) and ops.output_format in ('csv','csv+','shp'):
            raise(BlockedCombination)
        ## only perform calculation tests on subsetted regions
        if ops.geom is None and ops.calc is not None:
            raise(BlockedCombination)
            
    def check_exception(self,ii,kwds,e,tb):
        reraise = True
        if type(e) == AssertionError:
            ## nc files may not be clipped or aggregated
            if kwds['output_format'] == 'nc':
                if kwds['spatial_operation'] == 'clip' or kwds['aggregate'] is True:
                    reraise = False
        elif type(e) == NotImplementedError:
            ## groupings are required for calculations
            if kwds['calc'] is not None and kwds['calc_grouping'] is None:
                reraise = False
        if reraise:
            raise(CombinationError(ii,kwds))
        
    def execute(self):
        for combo in self: pass
    
    def get_parameters(self):
        ret = []
        for sc in itersubclasses(parms.AbstractParameter):
            if sc != parms.AbstractBooleanParameter:
                ret.append(sc)
        return(ret)
    
    
def main(pargs):
    import ipdb;ipdb.set_trace()
    
    
parser = argparse.ArgumentParser(description='combinatorial test runner for OCGIS')
parser.add_argument('-c','--combination',type=int,help='target start combination',default=0)
parser.set_defaults(func=main)

pargs = parser.parse_args()
pargs.func(pargs)
    
#if __name__ == '__main__':
#    nretries = 10
#    ctr_retry = 0
#    target = 2625
#    
#    def run_combos(target_combo=target):
#        cr = CombinationRunner(target_combo=target_combo)
#        cr.execute()
#    
#    while ctr_retry <= nretries:
#        try:
#            run_combos(target_combo=target)
#        except Exception as e:
#            ctr_retry += 1
#            if ctr_retry == nretries:
#                raise
#            else:
#                run_combos(target_combo=e.inumber)
