#from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
from util.helpers import get_temp_path
import netCDF4 as nc
from util.ncconv.experimental.ocg_meta import element, models
from util.ncconv.experimental.ocg_meta.element import PolyElementNotFound
from util.ncconv.experimental.helpers import itr_array, array_split
import numpy as np
from django.conf import settings
from multiprocessing import Manager
from multiprocessing.process import Process
from util.ncconv.experimental.pmanager import ProcessManager
from util.ncconv.experimental.ocg_converter.subocg_converter import SubOcgConverter


class NcConverter(SubOcgConverter):
    
#    def __init__(self,base_name,use_stat=False,meta=None,use_geom=False):
#        self.base_name = base_name
#        self.use_stat = use_stat
#        self.meta = meta
#        self.use_geom = use_geom
    
    def _convert_(self,ocg_dataset,has_levels=False,fill_value=1e20):
        
        if self.use_stat:
            sub = self.sub.sub
            substat = self.sub
        else:
            sub = self.sub
            substat = None
        
        print('starting convert...')
        ## create the dataset object
        path = get_temp_path(name=self.base_name,nest=True)
        tdataset = nc.Dataset(path,'w')
        try:
            ## return the grid dictionary
            grid = sub.to_grid_dict(ocg_dataset)
            ## initialize the element classes
            checks = []
            for check in element.PolyElement.get_checks():
                try:
                    obj = check(ocg_dataset.dataset)
                    checks.append(obj)
                except PolyElementNotFound:
                    if not has_levels:
                        pass
                    else:
                        raise
                except:
                    import ipdb;ipdb.set_trace()
            ## first do a loop over the dataset attributes
            for attr in ocg_dataset.dataset.ncattrs():
                captured = None
                for check in checks:
                    if check.name == attr and isinstance(check,element.DatasetPolyElement):
                        captured = check
                        break
                if captured is None:
                    calc = getattr(ocg_dataset.dataset,attr)
                else:
                    if isinstance(captured,element.SimpleTranslationalElement):
                        calc = captured.calculate()
                    elif isinstance(captured,element.SpatialTranslationalElement):
                        calc = captured.calculate(grid)
                    elif isinstance(captured,element.TemporalTranslationalElement):
                        calc = captured.calculate(sub.timevec)
                    elif isinstance(captured,models.FileName):
                        calc = self.base_name
                    else:
                        raise(ValueError)
                try:
                    setattr(tdataset,attr,calc)
                except:
                    ## need to account for unicode
                    setattr(tdataset,attr,str(calc))
            ## create the dimensions
            for dim in ocg_dataset.dataset.dimensions.keys():
                for check in checks:
                    if check.name == dim and isinstance(check,element.DimensionElement):
                        captured = check
                        break
                if isinstance(captured,element.TemporalDimensionElement):
                    if self.use_stat:
                        continue
                    calc = captured.calculate(sub.timevec)
                elif isinstance(captured,element.SpatialDimensionElement):
                    calc = captured.calculate(grid)
                elif isinstance(captured,element.LevelDimensionElement):
                    calc = captured.calculate(sub.levelvec)
                else:
                    raise(ValueError)
                tdataset.createDimension(captured.name,calc)
            ## create the variables
            for var in ocg_dataset.dataset.variables.keys():
                captured = None
                for check in checks:
                    if check.name == var and isinstance(check,element.VariablePolyElement):
                        captured = check
                        break
                if captured is None: continue
                if isinstance(captured,models.Row):
                    calc = captured.make_dimension_tup(models.LatitudeDimension(ocg_dataset.dataset))
                elif isinstance(captured,models.Column):
                    calc = captured.make_dimension_tup(models.LongitudeDimension(ocg_dataset.dataset))
                elif isinstance(captured,models.RowBounds):
                    calc = captured.make_dimension_tup(models.LatitudeDimension(ocg_dataset.dataset),
                                                       models.BoundsDimension(ocg_dataset.dataset))
                elif isinstance(captured,models.ColumnBounds):
                    calc = captured.make_dimension_tup(models.LongitudeDimension(ocg_dataset.dataset),
                                                       models.BoundsDimension(ocg_dataset.dataset))
                elif isinstance(captured,models.Time):
                    if self.use_stat:
                        continue
                    calc = captured.make_dimension_tup(models.TimeDimension(ocg_dataset.dataset))
                else:
                    raise
                tdataset.createVariable(captured.name,captured._dtype,calc)
                ## set the variable's data
                if isinstance(captured,element.TemporalTranslationalElement):
                    calc = captured.calculate(sub.timevec)
                elif isinstance(captured,element.SpatialTranslationalElement):
                    calc = captured.calculate(grid)
#                elif isinstance(captured,element.LevelDimensionElement):
#                    calc = captured.calculate(sub.levelvec)
                else:
                    raise(ValueError)
                tdataset.variables[captured.name][:] = calc
                ## set the variable's attrs
                for attr in ocg_dataset.dataset.variables[captured.name].ncattrs():
                    setattr(tdataset.variables[captured.name],attr,getattr(ocg_dataset.dataset.variables[captured.name],attr))
            ## set the actual value
            if self.use_stat:
                if has_levels:
                    raise(NotImplementedError)
                else:
                    ## these are the columns to exclude
                    exclude = ['ocgid','gid','level','geometry']
                    ## get the columns we want to write to the netcdf
                    cs = [c for c in substat.stats.keys() if c not in exclude]
                    ## loop through the columns and generate the numpy arrays to
                    ## to populate.
                    print('making variables...')
                    for ii,c in enumerate(cs):
                        ## get the correct python type from the column type
                        if type(substat.stats[c][0]) == float:
                            nctype = 'f4'
                        if type(substat.stats[c][0]) == int:
                            nctype = 'i4'
                        ## make the netcdf variable
                        tdataset.createVariable(c,nctype,('latitude','longitude'))
                    ## check for parallel
                    if settings.MAXPROCESSES > 1:
                        manager = Manager()
                        data = manager.list()
                        print('configuring processes...')
                        ## create the indices over which to split jobs
                        count = len(substat.stats['gid'])
                        indices = [[min(ary),max(ary)] 
                                   for ary in array_split(range(0,count+1),
                                                          settings.MAXPROCESSES)]
                        ## construct the processes
                        procs = [Process(target=self.f_fill,
                                         args=(data,rng,sub,substat,grid['gidx'].reshape(-1),cs))
                                 for rng in indices]
                        pmanager = ProcessManager(procs,settings.MAXPROCESSES)
                        ## run the processes
                        print('executing processes...')
                        pmanager.run()
                        ## reshape/transform list data into numpy arrays
                        ## the dictionary to hold merged data
                        merged = dict.fromkeys(data[0].keys(),np.zeros(len(grid['gidx'].reshape(-1))))
                        ## merge the data
                        for dd in data:
                            for key,value in dd.iteritems():
                                merged[key] = merged[key] + value
                        print('reformatting arrays...')
                        for key,value in merged.iteritems():
                            tary = value.reshape(len(grid['y']),len(grid['x']))
                            tary[grid['gidx'].mask] = fill_value
                            merged.update({key:tary})
                    else:
                        raise(NotImplementedError)
                    ## set the variable value in the nc dataset
                    for key,value in merged.iteritems():
                        tdataset.variables[key].missing_value = fill_value
                        tdataset.variables[key][:] = value
            else:
                gidx = grid['gidx']
                if has_levels:
                    raise(NotImplementedError)
                else:
                    value = np.empty((len(sub.timevec),len(grid['y']),len(grid['x'])),dtype=float)
                    for dt in sub.dim_time:
                        for ii,jj in itr_array(gidx):
                            if not hasattr(gidx[ii,jj],'mask'):
                                tgidx = gidx[ii,jj]
                                value[dt,ii,jj] = sub.value[dt,0,tgidx]
                            else:
                                value[dt,ii,jj] = fill_value
                    tdataset.createVariable('value','f4',('time','latitude','longitude'))
                tdataset.variables['value'].missing_value = fill_value
                tdataset.variables['value'][:] = value
            tdataset.sync()
            return(path)
        finally:
            tdataset.close()
            
    @staticmethod
    def f_fill(data,rng,sub,substat,gidx,cs):
        ## generate the arrays to fill
        fill = {}
        for cc in cs:
            fill.update({cc:np.zeros(len(gidx))})
        
        ## loop through the sub range
        for ii in range(*rng):
            idx = np.argmax(substat.stats['gid'][ii] == sub.gid)
            idx = np.argmax(gidx == idx)
            for key,value in fill.iteritems():
#                print ii,key,value,substat.stats[key][ii]
                value[idx] = substat.stats[key][ii]
        data.append(fill)
            
    def write(self,sub,ocg_dataset):
        return(self.convert(sub,ocg_dataset))
