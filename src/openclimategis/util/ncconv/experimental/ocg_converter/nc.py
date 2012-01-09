from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
from util.helpers import get_temp_path
import netCDF4 as nc
from util.ncconv.experimental.ocg_meta import element, models
from util.ncconv.experimental.ocg_meta.element import PolyElementNotFound
from util.ncconv.experimental.helpers import itr_array
import numpy as np


class NcConverter(OcgConverter):
    
    def _convert_(self,sub,ocg_dataset,has_levels=False,fill_value=1e20):
#        import ipdb;ipdb.set_trace()
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
                raise(NotImplementedError)
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
            
    def write(self,sub,ocg_dataset):
        return(self.convert(sub,ocg_dataset))
                
#    def _capture_(self,name):
#        ret = None
#        for check in cls._checks:
#            if check.name == name:
#                ret = check
#                break
#        return(ret)