import cProfile
from ocgis.api.operations import OcgOperations
from ocgis.api.iocg.interpreter_ocg import OcgInterpreter
import pstats


def run():
    dataset = {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc','variable':'tasmax'}
    ops = OcgOperations(dataset=dataset,snippet=True)
    ret = OcgInterpreter(ops).execute()
    
    
def main():
    path = '/tmp/stats'
    cProfile.run('run()',path)
    p = pstats.Stats(path)
#    p.sort_stats('time').print_stats(30)
    p.print_stats('get_numpy_data')
    import ipdb;ipdb.set_trace()
    
    
if __name__ == '__main__':
    main()