from util.ncconv.experimental.wrappers import multipolygon_operation
import datetime
import os
from osgeo import ogr
import warnings
from util.ncconv.experimental.ocg_converter import ShpConverter, KmlConverter,\
    CsvConverter, LinkedCsvConverter, LinkedShpConverter
import cProfile
import pstats
import sys
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, String, Float
from sqlalchemy.orm import relationship
import re
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import func
import time
sys.path.append('/home/bkoziol/Dropbox/UsefulScripts/python')
import shpitr


def filter_huron(feat):
    if feat['HUCNAME'] != 'HURON':
        return(False)
    else:
        return(True)

data = [
        dict(name='huron_huc8_watershed',
             path='/home/bkoziol/Dropbox/OpenClimateGIS/watersheds_4326.shp',
             filter=filter_huron,
             fields=['HUCNAME']),
#        dict(name='mi_huc8_watershed',
#             path='/home/bkoziol/Dropbox/OpenClimateGIS/watersheds_4326.shp',
#             filter=None,
#             fields=[]),
#        dict(name='state_boundaries',
#             path='/home/bkoziol/Dropbox/OpenClimateGIS/state_boundaries.shp',
#             filter=None,
#             fields=[])
        ]

def get_polygons(path,fields,filter):
    shp = shpitr.ShpIterator(path)
    polygons = []
    for feat in shp.iter_shapely(fields,filter=filter,as_multi=False):
        polygons.append(feat['geom'])
    return(polygons)

def f(polygons):
    times = 1
    for ii in range(0,times):
        print(ii+1)
        sub = multipolygon_operation('http://cida.usgs.gov/qa/thredds/dodsC/maurer/monthly',
                                     'sresa1b_miroc3-2-medres_2_Prcp',
                                     ocg_opts=dict(rowbnds_name='bounds_latitude',
                                                   colbnds_name='bounds_longitude',
                                                   calendar='proleptic_gregorian',
                                                   time_units='days since 1950-01-01 00:00:0.0'),
                                     polygons=polygons,
                                     time_range=[datetime.datetime(2011,11,1),datetime.datetime(2031,11,30)],
                                     level_range=None,
                                     clip=True,
                                     union=True,
                                     in_parallel=False,
                                     max_proc=8,
                                     max_proc_per_poly=1)
        assert(sub.value.shape[2] > 0)
        lshp = LinkedShpConverter(sub,'foo')
        out = lshp.convert(None)
        print(out)

def analyze(data,drop=False,load=False):
    
    engine = create_engine('sqlite:////home/bkoziol/tmp/profiling.sqlite')
    metadata = MetaData(bind=engine)
    Base = declarative_base(metadata=metadata)
    Session = sessionmaker(bind=engine)
    
    class SqlBase(object):
    
        @classmethod
        def get_or_create(cls,s,kwds,commit=False):
            qq = s.query(cls).filter_by(**kwds)
            try:
                obj = qq.one()
            except NoResultFound:
                obj = cls(**kwds)
                s.add(obj)
            if commit: s.commit()
            return(obj)
    
    class Scenario(SqlBase,Base):
        __tablename__ = 'scenario'
        sid = Column(Integer,primary_key=True)
        name = Column(String,nullable=False)
       
    class Function(SqlBase,Base):
        __tablename__ = 'function'
        fid = Column(Integer,primary_key=True)
        name = Column(String,nullable=False)
        
    class FileName(SqlBase,Base):
        __tablename__ = 'filename'
        fnid = Column(Integer,primary_key=True)
        name = Column(String,nullable=False)
        
        
    class Profile(SqlBase,Base):
        __tablename__ = 'profile'
        id = Column(Integer,primary_key=True)
        sid = Column(Integer,ForeignKey(Scenario.sid))
        fid = Column(Integer,ForeignKey(Function.fid))
        fnid = Column(Integer,ForeignKey(FileName.fnid),nullable=True)
        ncalls = Column(Integer,nullable=False)
        tottime = Column(Float,nullable=False)
        percall = Column(Float,nullable=False)
        cumtime = Column(Float,nullable=False)

        filename = relationship(FileName)
        scenario = relationship(Scenario)
        function = relationship(Function)
        
        def report(self,total):
            msg = [self.scenario.name]
            try:
                msg.append(self.filename.name)
            except AttributeError:
                msg.append('')
            msg.append(self.function.name)
            msg.append(str(self.tottime))
            msg.append(str(self.tottime/float(total)))
            msg.append(str(total))
            return(','.join(msg))
        
        @staticmethod
        def report_headers():
            return('scenario,filename,function,tottime,perctime,exetime')
    
    if load:
        if drop: metadata.drop_all(checkfirst=True)
        metadata.create_all(checkfirst=True)
        
        s = Session()
        for kwds in data:
            with open(os.path.join('/home/bkoziol/tmp',kwds['name']+'.txt'),'r') as out:
                txt = out.read()
            profiles = re.split('finished ::.*',txt)
            profiles = profiles[0:-1]
            for profile in profiles:
                profile = profile.strip()
                scenario_name = re.match('starting :: (.*)',profile).group(1)
                scenario = Scenario.get_or_create(s,dict(name=scenario_name))
                table = re.match('.*lineno\(function\)(.*)',profile,flags=re.DOTALL).group(1).strip()
                lines = re.split('\n',table)
                for line in lines:
                    line = line.strip()
        #            print line
                    elements = re.split(' {2,}',line)
                    if '{' in line and '}' in line:
                        filename = None
                    else:
                        try:
                            filename_name = re.match('.* (.*):.*',elements[4]).group(1)
                        except:
                            import ipdb;ipdb.set_trace()
                        filename = FileName.get_or_create(s,dict(name=filename_name))
                    rm = re.match('.*\((.*)\)|.*{(.*)}',elements[4])
                    if rm.group(1) is None:
                        function_name = rm.group(2)
                    else:
                        function_name = rm.group(1)
                    function = Function.get_or_create(s,dict(name=function_name))
                    obj = Profile()
                    obj.ncalls = elements[0]
                    obj.tottime = elements[1]
                    obj.percall = elements[2]
                    obj.cumtime = elements[3]
                    obj.filename = filename
                    obj.scenario = scenario
                    obj.function = function
                    s.add(obj)
        s.commit()
    else:
        s = Session()
        print(Profile.report_headers())
        for scenario in s.query(Scenario):
            ## get the total time
            total = s.query(func.sum(Profile.tottime)).filter_by(scenario=scenario)
            total = total.one()[0]
            ## get the top ten time things
            top = s.query(Profile).filter_by(scenario=scenario)
            top = top.order_by(Profile.tottime.desc())
            top = top.limit(10)
            for obj in top:
                print obj.report(total)
        import ipdb;ipdb.set_trace()

if __name__ == '__main__':
    for data_kwds in data:
        print('starting :: '+data_kwds['name'])
        polygons = get_polygons(data_kwds['path'],
                                data_kwds['fields'],
                                data_kwds['filter'])
        t1 = time.time()
        f(polygons)
#        cProfile.run('f(polygons)','/tmp/foo')
#        stats = pstats.Stats('/tmp/foo')
#        stats.sort_stats('time')
#        stats.strip_dirs()
#        stats.print_stats()
        print(time.time()-t1)
        print('finished :: '+data_kwds['name'])