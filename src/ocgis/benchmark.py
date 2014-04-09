import subprocess
import argparse
import logging
import time
import numpy as np
import datetime


## to get CPU info:
##     $ cat /proc/cpuinfo | grep 'model name'

'''
Intel(R) Core(TM) i7-2640M CPU @ 2.80GHz
commit 2e43c043ea3eda6eef71409c2cb472bc45f217c0
2014-04-09 14:47:18.687301
target: ocgis/test/test_simple/test_simple.py
     n: 10
  mean: 8.9257232666
 stdev: 0.0821190555739
   min: 8.78502821922
   max: 9.08388280869
59 tests
'''


def run_benchmark(test_target,n_iterations):
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    logger = logging.getLogger('benchmark')
    times = []
    for __ in range(n_iterations):
        logger.info('{0} of {1}'.format(__+1,n_iterations))
        t1 = time.time()
        subprocess.check_call(['nosetests',test_target])
        t2 = time.time()
        times.append(t2-t1)
    logger.info(datetime.datetime.now())
    logger.info('target: {0}'.format(test_target))
    logger.info('     n: {0}'.format(n_iterations))
    logger.info('  mean: {0}'.format(np.mean(times)))
    logger.info(' stdev: {0}'.format(np.std(times)))
    logger.info('   min: {0}'.format(np.min(times)))
    logger.info('   max: {0}'.format(np.max(times)))
        
def main():
    parser = argparse.ArgumentParser(description='Run OCGIS benchmark tests.')
    parser.add_argument('nosetest_target',help='Path to nosetest target.')
    parser.add_argument('--n_iterations','-n',default=10,type=int,help='Number of test iterations to run.')
    args = parser.parse_args()
    run_benchmark(args.nosetest_target,args.n_iterations)


if __name__ == '__main__':
    main()