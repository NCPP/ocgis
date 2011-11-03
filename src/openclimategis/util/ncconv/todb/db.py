from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String, DateTime
from sqlalchemy.orm import relationship


connstr = 'sqlite:///:memory:'
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
    datetime = Column(DateTime,unique=True,nullable=False,index=True)
    

class Meta(Base):
    __tablename__ = 'meta'
    mid = Column(Integer,primary_key=True)
    var_name = Column(String,unique=True,nullable=False)
    
    
class Value(Base):
    __tablename__ = 'value'
    id = Column(Integer,primary_key=True)
    gid = Column(ForeignKey(Geometry.gid))
    tid = Column(ForeignKey(Time.tid))
    mid = Column(ForeignKey(Meta.mid))
    value = Column(Float,nullable=False)
    level = Column(Integer,nullable=False)
    
    geometry = relationship(Geometry)
    time = relationship(Time)
    meta = relationship(Meta)