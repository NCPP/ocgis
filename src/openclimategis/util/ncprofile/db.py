from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String, Date
from sqlalchemy.orm import relationship
from geoalchemy.geometry import GeometryColumn, Polygon, GeometryDDL


#connstr = 'sqlite:///:memory:'
connstr = 'postgresql://bkoziol:BenK_84368636@localhost/test_ncprofile'
#connstr = 'sqlite:////tmp/foo.sql'
engine = create_engine(connstr)
metadata = MetaData(bind=engine)
Base = declarative_base(metadata=metadata)
Session = sessionmaker(bind=engine)


class Dataset(Base):
    __tablename__ = 'nc_dataset'
    id = Column(Integer,primary_key=True)
    uri = Column(String)


class Variable(Base):
    __tablename__ = 'nc_variable'
    id = Column(Integer,primary_key=True)
    dataset_id = Column(ForeignKey(Dataset.id))
#    category = Column(String)
    code = Column(String)
#    dimensions = Column(String)
    ndim = Column(Integer)
#    shape = Column(String)
    
    dataset = relationship(Dataset,backref=__tablename__)


class Attribute(Base):
    __tablename__ = 'nc_attr'
    id = Column(Integer,primary_key=True)
    variable_id = Column(ForeignKey(Variable.id))
    code = Column(String)
    value = Column(String)

    variable = relationship(Variable,backref=__tablename__)


class Dimension(Base):
    __tablename__ = 'nc_dimension'
    id = Column(Integer,primary_key=True)
    variable_id = Column(ForeignKey(Variable.id))
    code = Column(String)
    index = Column(Integer)

    variable = relationship(Variable,backref=__tablename__)
    
    
class IndexBase(object):
    id = Column(Integer,primary_key=True)
    index = Column(Integer)
    
    @declared_attr
    def dimension_id(self):
        return(Column(ForeignKey(Dimension.id)))
    
    @declared_attr
    def dimension(self):
        return(relationship(Dimension,backref=self.__tablename__))
    
    
class IndexTime(IndexBase,Base):
    __tablename__ = 'nc_index_time'
    value = Column(Date)
    
    
class IndexSpatial(IndexBase,Base):
    __tablename__ = 'nc_index_spatial'
    value = GeometryColumn(Polygon)
GeometryDDL(IndexSpatial.__table__)
    

try:
    metadata.drop_all()
finally:
    metadata.create_all()