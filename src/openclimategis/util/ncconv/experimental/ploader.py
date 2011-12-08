from multiprocessing.process import Process
import numpy as np
from sqlalchemy.exc import OperationalError
from multiprocessing import Lock


class ParallelLoader(object):
    
    def __init__(self,procs=1):
        self.procs = procs
        
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
        if self.procs == 1:
            self.loader(pmodel)
        elif self.procs > 1:
            pmodels = pmodel.split(self.procs)
            self.load_models(pmodels)
        else:
            raise(ValueError('"procs" must be one or greater.'))
        
    def load_models(self,pmodels):
        """assumes the pmodels length is equivalent to the number of desired processes."""
        lock = Lock()
        processes = [Process(target=self.loader,args=(pm,lock)) for pm in pmodels]
        self.run(processes)
        
    def run(self,processes):
        for ii,process in enumerate(processes,start=1):
            process.start()
            if ii >= self.procs:
                while sum([p.is_alive() for p in processes]) >= self.procs:
                    pass
        for p in processes:
            p.join() 
        

class ParallelModel(object):
    
    def __init__(self,Model,pvars):
        self.Model = Model
        self.pvars = pvars
        
    def iter_data(self,as_dict=False):
        idx = 0
        while True:
            try:
                attrs = dict(zip([pvar.name for pvar in self.pvars],
                                 [pvar.get_data(idx) for pvar in self.pvars]))
                if as_dict:
                    ret = attrs
                else:
                    ret = self.Model(**attrs)
                idx += 1
                yield(ret)
            except IndexError:
                raise(StopIteration)
            
    def split(self,n):
        npvars = [pvar.split(n) for pvar in self.pvars]
        pmodels = []
        for ii in range(0,len(npvars[0])):
            pmodels.append(ParallelModel(self.Model,[npvar[ii] for npvar in npvars]))
        return(pmodels)
        
        
class ParallelVariable(object):
    
    def __init__(self,name,data=[],op=None):
        self.name = name
        self.data = data
        self.op = op
        
    def __len__(self):
        return(len(self.data))
        
    def __iter__(self):
        for d in self.data:
            if self.op is not None:
                d = self.op(d)
            yield({self.name:d})
            
    def append(self,val):
        self.data = self.data + [val]
            
    def get_data(self,idx):
        if self.op is not None:
            d = self.op(self.data[idx])
        else:
            d = self.data[idx]
        return(d)
            
    def split(self,n):
        datas = np.array_split(self.data,n)
        return([ParallelVariable(self.name,data.tolist(),op=self.op)
                 for data in datas])

### geometry
#point_data = ['point1','point2','point3','point4']
#wkt = ParallelVariable('wkt',point_data)
#wkb = ParallelVariable('wkb',point_data,op=(lambda x: x.upper()))
#geometry = ParallelModel(Geometry,[wkb,wkt])
#
### value
#gid = ParallelVariable('gid',[1,2,1,2])
#level = ParallelVariable('level',[1,1,1,1])
#val = ParallelVariable('value',[3,4,5,6])
#value = ParallelModel(Value,[gid,level,val])
#
#ploader = ParallelLoader(Session,procs=4)
#for model in [geometry,value]:
#    ploader.load_model(model)