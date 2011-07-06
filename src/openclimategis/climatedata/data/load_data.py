import os
from climatedata.models import ClimateModel, Dataset, AttributeDataset,\
    IndexTime, Variable, Dimension, AttributeVariable, IndexSpatial
import netCDF4
import warnings
from shapely.geometry.point import Point
from django.contrib.gis.geos.geometry import GEOSGeometry
from shapely.geometry.polygon import Polygon
from django.db import transaction



@transaction.commit_on_success()
def load_climatemodel(archive,uris,**kwds):
    ncm = NcModelImporter(uris,**kwds)
    ncm.load(archive)


class NcModelImporter(object):
    """
    Load a climate model with URIs pointing to model output datasets.
    
    code -- short name of the climate model. i.e. 'bccr_bcm2.0'.
    uris -- list of URIs pointing to model output datasets. each URI should return
      a Dataset instance when used by netCDF4.Dataset().
    """
    
    def __init__(self,uris,**kwds):
        self.uris = uris
        self.name = kwds.get('name')
        self.code = kwds.get('code')
        self.url = kwds.get('url')
        
    def load(self,archive):
        """
        Load the model into the database using a foreign Archive object.
        
        archive -- Archive database object.
        """
        
        kwds = dict(archive=archive,
                    name=self.name,
                    code=self.code,
                    url=self.url)
        cm,created = ClimateModel.objects.get_or_create(**kwds)
#        cm = db.ClimateModel(code=self.code,archive=archive)
#        s.add(cm)
        first = True
        for uri in self.uris:
            if first:
                spatial = True
                first = False
            else:
                spatial = False
            ndp = NcDatasetImporter(uri)
            ndp.load(cm,spatial=spatial)
#        s.commit()
        

class NcDatasetImporter(object):
    """
    Import a netCDF4 Dataset object into the database.
    """
    
    def __init__(self,uri):
        self.uri = uri
        self.dataset = netCDF4.Dataset(uri,'r')
        
        self._set_name_('time',['time'])
        self._set_name_('time_bnds',['time_bnds','time_bounds'])
        self._set_name_('row',['lat','latitude'])
        self._set_name_('col',['lon','longitude'])
        self._set_name_('row_bnds',['lat_bounds','latitude_bounds','lat_bnds','latitude_bnds','bounds_latitude'])
        self._set_name_('col_bnds',['lon_bounds','longitude_bounds','lon_bnds','longitude_bnds','bounds_longitude'])
    
    def _set_name_(self,target,options):
        "Search naming options for target variables."
        
        ret = None
        for key in self.dataset.variables.keys():
            if key in options:
                ret = key
                break
        setattr(self,target,ret)
        if not ret:
            warnings.warn('variable "{0}" not found in {1}. setting to "None" and no load is attempted.'.format(target,self.dataset.variables.keys()))
        
    def load(self,cm,spatial=True,temporal=True):
        print('loading '+self.uri+'...')
        print('loading dataset...')
        dataset = self._dataset_(cm)
        print('loading variables...')
        self._variables_(dataset)
        if spatial:
            print('loading spatial grid...')
            self._spatial_(cm)
        if temporal:
            print('loading temporal grid...')
            self._temporal_(dataset)
#        s.commit()
        print('success.')
        
    def _dataset_(self,climatemodel):
        obj = Dataset()
        obj.climatemodel = climatemodel
        obj.name = os.path.split(self.uri)[1]
        obj.uri = self.uri
        obj.save()
#        s.add(obj)
        for attr in self.dataset.ncattrs():
            value = getattr(self.dataset,attr)
            a = AttributeDataset()
            a.dataset = obj
            a.name = attr
            a.value = str(value)
            a.save()
#            s.add(a)
#        s.commit()
        return(obj)
    
    def _temporal_(self,dataset):
        value = self.dataset.variables[self.time]
        vec = netCDF4.num2date(value[:],value.units,value.calendar)
        for ii in xrange(len(vec)):
            idx = IndexTime()
            idx.dataset = dataset
            idx.index = ii
            idx.value = vec[ii]
            if self.time_bnds:
                bounds = netCDF4.num2date(self.dataset.variables[self.time_bnds][ii,:],
                                     value.units,
                                     value.calendar)
                idx.lower = bounds[0]
                idx.upper = bounds[1]
#            s.add(idx)
            idx.save()
    
    def _variables_(self,dataset):
        for key,value in self.dataset.variables.iteritems():
#            q = s.query(db.Variable).filter(db.Variable.climatemodel==climatemodel)\
#                                    .filter(db.Variable.code==key)\
#                                    .filter(db.Variable.ndim==value.ndim)
#            try:
#                obj = q.one()
#            except NoResultFound:
            obj = Variable()
            obj.dataset = dataset
            obj.name = key
    #            obj.dimensions = str(value.dimensions)
            obj.ndim = value.ndim
            obj.save()
#            obj.shape = str(value.shape)

            ## TIME DIMENSION -------------------------------------------------
            
            ## construct the dimensions
            for dim,sh in zip(value.dimensions,value.shape):
#                d = db.Dimension()
                d = Dimension()
                d.variable = obj
                d.name = dim
                d.size = sh
                d.position = value.dimensions.index(dim)
#                s.add(d)
                d.save()
            
#            ## classify the variable
#            obj.category = 'variable' ## the default
#            for key1,value1 in CLASS.iteritems():
#                if key1 == 'bounds':
#                    for v in value1:
#                        if v in key:
#                            obj.category = key1
#                            break
#                else:
#                    if key in value1:
#                        obj.category = key1
#                        break
            
#            s.add(obj)
            for ncattr in value.ncattrs():
                a = AttributeVariable()
                a.variable = obj
                a.name = ncattr
                a.value = str(getattr(value,ncattr))
#                s.add(a)
                a.save()
#            import ipdb;ipdb.set_trace()

#    @transaction.commit_on_success()
    def _spatial_(self,climatemodel):
        col = self.dataset.variables[self.col][:]
        row = self.dataset.variables[self.row][:]
        col_bnds = self.dataset.variables[self.col_bnds][:]
        row_bnds = self.dataset.variables[self.row_bnds][:]
#        geoms = []
        total = len(col)*len(row)
        ctr = 0
        for ii in xrange(len(col)):
            for jj in xrange(len(row)):
                if ctr%2000 == 0:
                    print('  {0} of {1}'.format(ctr,total))
                ctr += 1
#                obj = db.IndexSpatial()
                obj = IndexSpatial()
                obj.row = ii
                obj.col = jj
                obj.climatemodel = climatemodel
                pt = Point(col[ii],row[jj])
                obj.centroid = GEOSGeometry(str(pt))
                poly = Polygon(((col_bnds[ii,0],row_bnds[jj,0]),
                               (col_bnds[ii,0],row_bnds[jj,1]),
                               (col_bnds[ii,1],row_bnds[jj,1]),
                               (col_bnds[ii,1],row_bnds[jj,0])))
                obj.geom = GEOSGeometry(str(poly))
                obj.save()
#                geoms.append(obj)
#                s.add(obj)
#        climatemodel.indexspatial = geoms
#        print('  committing...')
#        s.commit()
#                pass

        
    def __del__(self):
        try:
            self.dataset.close()
        except:
            pass