import os

import netCDF4 as nc
from ocgis.test.make_test_data_subset import subset_first_two_years, \
    SingleYearFile
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.schema import Column, UniqueConstraint, ForeignKey
from sqlalchemy.types import Integer, String, DateTime, Float

import ocgis

Base = declarative_base()


class AbstractTestData(object):
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    variable = Column(String, nullable=False)
    time_start = Column(DateTime, nullable=False)
    time_end = Column(DateTime, nullable=False)
    time_interval_days = Column(Float, nullable=False)
    storage_location = Column(String, nullable=False)

    @property
    def full_path(self):
        return (os.path.join(self.storage_location, self.filename))

    def set_time(self):
        rd = ocgis.RequestDataset(uri=self.full_path, variable=self.variable)
        temporal = rd.ds.temporal
        self.time_start = temporal.value[0]
        self.time_end = temporal.value[-1]
        self.time_interval_days = temporal.resolution


class OriginalTestData(AbstractTestData, Base):
    __tablename__ = 'original'
    __table_args__ = (UniqueConstraint('filename'),)
    group = Column(String)

    def __init__(self, dirpath, filename):
        self.storage_location = dirpath
        self.filename = filename
        ## make a guess at the target variables
        ds = nc.Dataset(self.full_path, 'r')
        try:
            for key, value in ds.variables.iteritems():
                if len(value.dimensions) > 2 and value.dimensions[0] == 'time':
                    self.variable = key
                    break
            self.set_time()
        finally:
            ds.close()


class SubsetTestData(AbstractTestData, Base):
    __tablename__ = 'subset'
    __table_args__ = (UniqueConstraint('filename'),)
    id_original = Column(Integer, ForeignKey(OriginalTestData.id))
    original = relationship(OriginalTestData, backref="subset")

    def __init__(self, otd, subset_path, subset_prefix):
        self.original = otd
        new_file_name = subset_prefix + '_' + otd.filename
        out_nc = os.path.join(subset_path, new_file_name)
        try:
            subset_first_two_years(otd.full_path, out_nc)
            new_path = subset_path
        except SingleYearFile:
            new_file_name = otd.filename
            new_path = otd.storage_location
        self.storage_location = new_path
        self.filename = new_file_name
        self.variable = otd.variable
        self.set_time()
