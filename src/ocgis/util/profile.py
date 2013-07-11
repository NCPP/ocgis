import cProfile
from ocgis.api.operations import OcgOperations
import pstats
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData, Column
from sqlalchemy.ext.declarative.api import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String
from tempfile import NamedTemporaryFile, mkdtemp
import sys
import re
import os
import shutil


#connstr = 'sqlite://'
## four slashes for absolute paths - three for relative
connstr = 'sqlite:///{0}'.format('/tmp/profile.sqlite')
engine = create_engine(connstr)
metadata = MetaData(bind=engine)
Base = declarative_base(metadata=metadata)
Session = sessionmaker(bind=engine)


class Results(Base):
    __tablename__ = 'results'
    rid = Column(Integer,primary_key=True)
    ncalls = Column(Integer)
    tottime = Column(Float)
    filename = Column(String)
    percall_tottime = Column(Float)
    cumtime = Column(Float)
    percall_cumtime = Column(Float)
    
    
class ExecutionInfo(Base):
    __tablename__ = 'info'
    eid = Column(Integer,primary_key=True)
    timestamp = Column(String)
    timing = Column(String)


def run():
#    for ii in range(10):
#        dir_output = mkdtemp()
#        try:
#            dataset = {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc','variable':'tasmax'}
#            ops = OcgOperations(dataset=dataset,output_format='csv+',geom='state_boundaries',select_ugid=[25,26],
#                                dir_output=dir_output,calc=[{'func':'mean','name':'mean'}],calc_grouping=['month'])
#            ret = ops.execute()
#        finally:
#            shutil.rmtree(dir_output)
    sys.path.append('/home/local/WX/ben.koziol/links/git/examples')
    from narccap import city_centroid_subset
    city_centroid_subset.main()
    
    
def main():
    try:
        os.remove('/tmp/profile.sqlite')
    except:
        pass
    metadata.create_all()
    f = NamedTemporaryFile(delete=False)
    prev_stdout = sys.stdout
    sys.stdout = f
    path = '/tmp/stats'
    cProfile.run('run()',path)
    p = pstats.Stats(path)
    p.print_stats()
    f.close()
    sys.stdout = prev_stdout
    with open(f.name,'r') as dstats:
        lines = dstats.readlines()
#    p.sort_stats('time').print_stats(30)
#    p.print_stats('get_numpy_data')
    start = 6
    build = True
    s = Session()
    while True:
        try:
            line = lines[start].strip()
        except IndexError:
            break
        if build:
            s.add(ExecutionInfo(timestamp=lines[0].strip(),timing=lines[2].strip()))
            headers = re.split(' +',line)
            headers[-1] = 'filename'
            headers[2] = 'percall_tottime'
            headers[4] = 'percall_cumtime'
            build = False
        else:
            parts = re.split(' {2,5}',line)
            data = dict(zip(headers,parts))
            if not 'filename' in data:
                try:
                    percall_cumtime,filename = re.split('[0-9] ',data['percall_cumtime'],maxsplit=1)
                    data['percall_cumtime'] = percall_cumtime
                    data['filename'] = filename
                except KeyError:
                    if line == '':
                        start += 1
                        continue
                    else:
                        raise
            results = Results(**data)
            s.add(results)
        start += 1
    s.commit()
    s.close()
    
    
if __name__ == '__main__':
    main()