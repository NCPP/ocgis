from tempfile import mkstemp
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative.api import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import String, Integer
from sqlalchemy.orm import relationship


fd,db_path = mkstemp(suffix='.sqlite')
connstr = 'sqlite:///{0}'.format(db_path)
engine = create_engine(connstr)
metadata = MetaData(bind=engine)
Base = declarative_base(metadata=metadata)
Session = sessionmaker(bind=engine)


class Category(Base):
    __tablename__ = 'category'
    __table_args__ = (UniqueConstraint('key','label'),)
    cid = Column(Integer,primary_key=True)
    key = Column(String)
    label = Column(String,nullable=False)
    
    download_url = Column(String,nullable=True)
    metadata_url = Column(String,nullable=True)
    download_date = Column(String,nullable=True)
    description = Column(String,nullable=True)
    history = Column(String,nullable=True)
    
    
class Subcategory(Base):
    __tablename__ = 'subcategory'
    __table_args__ = (UniqueConstraint('cid','label'),)
    sid = Column(Integer,primary_key=True)
    cid = Column(Integer,ForeignKey(Category.cid))
    label = Column(String,nullable=False)
    
    category = relationship(Category,backref="subcategory")
    
    
class Geometry(Base):
    __tablename__ = 'geometry'
    __table_args__ = (UniqueConstraint('cid','sid','label','ugid'),)
    gid = Column(Integer,primary_key=True)
    cid = Column(Integer,ForeignKey(Category.cid),nullable=False)
    sid = Column(Integer,ForeignKey(Subcategory.sid),nullable=True)
    ugid = Column(Integer,nullable=False)
    envelope = Column(String,nullable=False)
    label = Column(String,nullable=False)
    
    category = relationship(Category,backref='geometry')
    subcategory = relationship(Subcategory,backref='geometry')
