from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String, DateTime, Date
from sqlalchemy.orm import relationship


connstr = 'sqlite://'
#connstr = 'postgresql://bkoziol:<password>@localhost/<database>'
#connstr = 'postgresql://{user}:{password}@{host}/{database}'
engine = create_engine(connstr)
metadata = MetaData(bind=engine)
Base = declarative_base(metadata=metadata)
Session = sessionmaker(bind=engine)


class Geometry(Base):
    __tablename__ = 'geometry'
    gid = Column(Integer,primary_key=True)
    wkt = Column(String,unique=True,nullable=False)
    
    
class Time(Base):
    __tablename__ = 'time'
    tid = Column(Integer,primary_key=True)
    time = Column(DateTime,unique=True,nullable=False,index=True)


class Value(Base):
    __tablename__ = 'value'
    ocgid = Column(Integer,primary_key=True)
    gid = Column(ForeignKey(Geometry.gid))
    tid = Column(ForeignKey(Time.tid))
    level = Column(Integer,nullable=False)
    value = Column(Float,nullable=False)
    
    geometry = relationship(Geometry)
    time = relationship(Time)
    
    def __repr__(self):
        msg = ['geometry={0}'.format(self.geometry.wkt[0:7])]
        msg.append('datetime={0}'.format(self.time.dt))
        msg.append('level={0}'.format(self.level))
        msg.append('value={0}'.format(self.value))
        return(','.join(msg))
    

metadata.create_all()