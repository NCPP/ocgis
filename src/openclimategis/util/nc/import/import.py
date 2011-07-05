import netCDF4 as nc
import os
import db
import sys
from shapely.geometry.point import Point
from geoalchemy.base import WKTSpatialElement
from shapely.geometry.polygon import Polygon
import warnings


class NcModelImporter(object):
    """
    Load a climate model with URIs pointing to model output datasets.
    
    code -- short name of the climate model. i.e. 'bccr_bcm2.0'.
    uris -- list of URIs pointing to model output datasets. each URI should return
      a Dataset instance when used by netCDF4.Dataset().
    """
    
    def __init__(self,code,uris):
        self.code = code
        self.uris = uris
        
    def load(self,s,archive):
        """
        Load the model into the database using a foreign Archive object.
        
        s -- Session object.
        archive -- Archive database object.
        """
        
        cm = db.ClimateModel(code=self.code,archive=archive)
        s.add(cm)
        first = True
        for uri in self.uris:
            if first:
                spatial = True
                first = False
            else:
                spatial = False
            ndp = NcDatasetImporter(uri)
            ndp.load(s,cm,spatial=spatial)
        s.commit()
        

class NcDatasetImporter(object):
    """
    Import a netCDF4 Dataset object into the database.
    """
    
    def __init__(self,uri):
        self.uri = uri
        self.dataset = nc.Dataset(uri,'r')
        
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
        
    def load(self,s,cm,spatial=True,temporal=True):
        print('loading '+self.uri+'...')
        print('loading dataset...')
        dataset = self._dataset_(s,cm)
        print('loading variables...')
        self._variables_(s,dataset)
        if spatial:
            print('loading spatial grid...')
            self._spatial_(s,cm)
        if temporal:
            print('loading temporal grid...')
            self._temporal_(s,dataset)
        s.commit()
        print('success.')
        
    def _dataset_(self,s,climatemodel):
        obj = db.Dataset()
        obj.climatemodel = climatemodel
        obj.code = os.path.split(self.uri)[1]
        obj.uri = self.uri
        s.add(obj)
        for attr in self.dataset.ncattrs():
            value = getattr(self.dataset,attr)
            a = db.AttributeDataset()
            a.dataset = obj
            a.code = attr
            a.value = str(value)
            s.add(a)
        s.commit()
        return(obj)
    
    def _temporal_(self,s,dataset):
        value = self.dataset.variables[self.time]
        vec = nc.num2date(value[:],value.units,value.calendar)
        for ii in xrange(len(vec)):
            idx = db.IndexTime()
            idx.dataset = dataset
            idx.index = ii
            idx.value = vec[ii]
            if self.time_bnds:
                bounds = nc.num2date(self.dataset.variables[self.time_bnds][ii,:],
                                     value.units,
                                     value.calendar)
                idx.lower = bounds[0]
                idx.upper = bounds[1]
            s.add(idx)
    
    def _variables_(self,s,dataset):
        for key,value in self.dataset.variables.iteritems():
#            q = s.query(db.Variable).filter(db.Variable.climatemodel==climatemodel)\
#                                    .filter(db.Variable.code==key)\
#                                    .filter(db.Variable.ndim==value.ndim)
#            try:
#                obj = q.one()
#            except NoResultFound:
            obj = db.Variable()
            obj.dataset = dataset
            obj.code = key
    #            obj.dimensions = str(value.dimensions)
            obj.ndim = value.ndim
#            obj.shape = str(value.shape)

            ## TIME DIMENSION -------------------------------------------------
            
            ## construct the dimensions
            for dim,sh in zip(value.dimensions,value.shape):
                d = db.Dimension()
                d.variable = obj
                d.code = dim
                d.size = sh
                d.position = value.dimensions.index(dim)
                s.add(d)
            
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
            
            s.add(obj)
            for ncattr in value.ncattrs():
                a = db.AttributeVariable()
                a.variable = obj
                a.code = ncattr
                a.value = str(getattr(value,ncattr))
                s.add(a)
#            import ipdb;ipdb.set_trace()

    def _spatial_(self,s,climatemodel):
        col = self.dataset.variables[self.col][:]
        row = self.dataset.variables[self.row][:]
        col_bnds = self.dataset.variables[self.col_bnds][:]
        row_bnds = self.dataset.variables[self.row_bnds][:]
        geoms = []
        total = len(col)*len(row)
        ctr = 0
        for ii in xrange(len(col)):
            for jj in xrange(len(row)):
                if ctr%2000 == 0:
                    print('  {0} of {1}'.format(ctr,total))
                ctr += 1
                obj = db.IndexSpatial()
                obj.row = ii
                obj.col = jj
#                obj.dataset = dataset
                pt = Point(col[ii],row[jj])
                obj.centroid = WKTSpatialElement(str(pt))
                poly = Polygon(((col_bnds[ii,0],row_bnds[jj,0]),
                               (col_bnds[ii,0],row_bnds[jj,1]),
                               (col_bnds[ii,1],row_bnds[jj,1]),
                               (col_bnds[ii,1],row_bnds[jj,0])))
                obj.geom = WKTSpatialElement(str(poly))
                geoms.append(obj)
#                s.add(obj)
        climatemodel.indexspatial = geoms
        print('  committing...')
        s.commit()
#                pass

        
    def __del__(self):
        try:
            self.dataset.close()
        except:
            pass
        

if __name__ == '__main__':
#    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
#    s = db.Session()
#    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
#    ncp = NcDatasetImporter(uri)
#    ncp.load(s,1)
#    s.close()
#    import ipdb;ipdb.set_trace()

    s = db.Session()
    d = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data'
#    climatemodels = ['bccr_bcm2.0','bccr_bcm2.0']
    for root,dirs,files in os.walk(d):
#        dirs.sort()
        for d in dirs:
#            for f in os.listdir(os.path.join(root,d)):
            uris = [os.path.join(root,d,f) for f in os.listdir(os.path.join(root,d)) if f.endswith('.nc')]
#            models = []
#            for uri in uris:
#                models.append(uri.split('.')[0])
            archive = db.Archive(code=d)
            ncm = NcModelImporter('bccr_bcm2.0',uris)
            ncm.load(s,archive)
            sys.exit()
#                if f.endswith('.nc'):
#                    uri = os.path.join(root,f)
#                    print d,uri
#                print(uri)
#                ncp = NcDatasetImporter(uri)
#                ncp.load()
    s.close()
#                sys.exit()
