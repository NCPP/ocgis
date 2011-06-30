from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String, Date, DateTime
from sqlalchemy.orm import relationship
from geoalchemy.geometry import GeometryColumn, Polygon, GeometryDDL, Point


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
    code = Column(String)
    uri = Column(String)
    
    
class AttributeDataset(Base):
    __tablename__ = 'nc_attr_dataset'
    id = Column(Integer,primary_key=True)
    dataset_id = Column(ForeignKey(Dataset.id))
    code = Column(String)
    value = Column(String)

    dataset = relationship(Dataset,backref=__tablename__)


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


class AttributeVariable(Base):
    __tablename__ = 'nc_attr_variable'
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
    index_name = Column(String)

    variable = relationship(Variable,backref=__tablename__)
    
    
class IndexBase(object):
    id = Column(Integer,primary_key=True)
    
    @declared_attr
    def dataset_id(self):
        return(Column(ForeignKey(Dataset.id)))
    
    @declared_attr
    def dataset(self):
        return(relationship(Dataset,backref=self.__tablename__))
    
    
class IndexTime(IndexBase,Base):
    __tablename__ = 'nc_index_time'
    index = Column(Integer)
    lower = Column(DateTime)
    value = Column(DateTime)
    upper = Column(DateTime)
    
    
class IndexSpatial(IndexBase,Base):
    __tablename__ = 'nc_index_spatial'
    id = Column(Integer,primary_key=True)
    dataset_id = Column(ForeignKey(Dataset.id))
    row = Column(Integer)
    col = Column(Integer)
    geom = GeometryColumn(Polygon)
    centroid = GeometryColumn(Point)
    
    dataset = relationship(Dataset,backref=__tablename__)
GeometryDDL(IndexSpatial.__table__)
    

try:
    metadata.drop_all()
finally:
    metadata.create_all()