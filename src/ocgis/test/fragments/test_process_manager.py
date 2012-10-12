import unittest
import time
from multiprocessing import Pool
import numpy as np
import itertools


def iter_proc_args():
    for ii in range(0,10):
        yield(ii)
        
def operation(ii):
        time.sleep(np.random.randint(0,3))
        return(ii*2)


class SubsetOperation(object):
    
    def __init__(self,it_procs,serial=True,nprocs=1):
        self.it_procs = it_procs
        self.serial = serial
        self.nprocs = nprocs
        
    def __iter__(self):
        if self.serial:
            it = itertools.imap(operation,self.it_procs())
        else:
            pool = Pool(processes=self.nprocs)
            it = pool.imap_unordered(operation,self.it_procs())
        while True:
            try:
                yield(it.next())
            except StopIteration:
                break
    
    def run(self):
        path = '/tmp/foo.txt'
        with open(path,'w') as f:
            for value in self:
                f.write(str(value))
        return(path)
        

class TestProcessManager(unittest.TestCase):

    def test(self):
        serial = False
        conv = SubsetOperation(iter_proc_args,serial=serial,nprocs=4)
        ret = conv.run()
        print ret
                
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()