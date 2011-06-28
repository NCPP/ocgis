from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String
from sqlalchemy.orm import relationship


#connstr = 'sqlite:///:memory:'
#connstr = 'postgresql://bkoziol:<password>@localhost/<database>'
connstr = 'sqlite:////tmp/foo.sql'
engine = create_engine(connstr)
metadata = MetaData(bind=engine)
Base = declarative_base(metadata=metadata)
Session = sessionmaker(bind=engine)


class NetCDF(Base):
    __tablename__ = 'netcdf'
    id = Column(Integer,primary_key=True)
    nc = Column(String)


class Variable(Base):
    __tablename__ = 'variable'
    id = Column(Integer,primary_key=True)
    netcdf_id = Column(ForeignKey(NetCDF.id))
    variable = Column(String)
    dimensions = Column(String)
    ndim = Column(Integer)
    shape = Column(String)
    
    netcdf = relationship(NetCDF,backref=__tablename__)


class Attributes(Base):
    __tablename__ = 'attr'
    id = Column(Integer,primary_key=True)
    variable_id = Column(ForeignKey(Variable.id))
    attr_name = Column(String)
    value = Column(String)


try:
    metadata.drop_all()
finally:
    metadata.create_all()