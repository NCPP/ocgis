from multiprocessing.process import Process
from multiprocessing import Lock
from util.ncconv.experimental.helpers import array_split


class ParallelLoader(object):
    
    def __init__(self,procs=1,use_lock=True):
        self.procs = procs
        self.use_lock = use_lock
        
    @staticmethod
    def loader(pmodel,lock=None):
        coll = []
        for ii,attrs in enumerate(pmodel.iter_data(as_dict=True)):
            if ii == 0:
                i = pmodel.Model.__table__.insert()
            coll.append(attrs)
        if lock is None:
            i.execute(*coll)
        else:
            lock.acquire()
            i.execute(*coll)
            lock.release()
        
    def load_model(self,pmodel):
        if self.procs > 1:
            pmodels = pmodel.split(self.procs)
            self.load_models(pmodels)
        else:
            self.loader(pmodel)
        
    def load_models(self,pmodels):
        """assumes the pmodels length is equivalent to the number of desired processes."""
        if self.use_lock:
            lock = Lock()
        else:
            lock = None
        processes = [Process(target=self.loader,args=(pm,lock)) for pm in pmodels]
        self.run(processes)
        
    def run(self,processes):
        for process in processes: process.start()
        for process in processes: process.join()
        

class ParallelModel(object):
    
    def __init__(self,Model,data):
        self.Model = Model
        self.data = data
        
    def __len__(self):
        return(len(self.data[self.data.keys()[0]]))
        
    def iter_data(self,as_dict=False):
        idx = 0
        while True:
            try:
                attrs = dict([[key,value[idx]] for key,value in self.data.iteritems()])
                if as_dict:
                    ret = attrs
                else:
                    ret = self.Model(**attrs)
                idx += 1
                yield(ret)
            except IndexError:
                raise(StopIteration)

    def split(self,n):
        storage = []
        ref_idx = range(0,len(self.data[self.data.keys()[0]]))
        groups = array_split(ref_idx,n)
        for group in groups:
            data = dict([[key,[value[g] for g in group]] for key,value in self.data.iteritems()])
            storage.append(ParallelModel(self.Model,data))
        return(storage)
    
    
class ParallelGenerator(ParallelLoader):
    
    def __init__(self,Model,indices,f,fkwds={},procs=1,use_lock=True):
        self.Model = Model
        self.indices = indices
        self.f = f
        self.fkwds = fkwds
        self.procs = procs
        self.use_lock = use_lock
        
    @staticmethod
    def loader(indices,f,fkwds,Model,lock=None):
        coll = []
        for ii,jj in enumerate(indices):
            if ii == 0:
                i = Model.__table__.insert()
            coll.append(f(jj,**fkwds))
        if lock is None:
            i.execute(*coll)
        else:
            lock.acquire()
            i.execute(*coll)
            lock.release()
        
    def load(self):
        indices_groups = array_split(self.indices,self.procs)
        if self.use_lock:
            lock = Lock()
        else:
            lock = None
        processes = [Process(target=self.loader,args=(indices_group,self.f,self.fkwds,self.Model,lock))
                     for indices_group in indices_groups]
        self.run(processes)
