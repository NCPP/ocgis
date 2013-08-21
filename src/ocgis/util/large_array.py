import ocgis
from ocgis.calc import tile
import netCDF4 as nc
from ocgis.util.helpers import ProgressBar
from ocgis.interface.nc.dataset import NcDataset
from ocgis.api.collection import CalcCollection, MultivariateCalcCollection
from ocgis.api.request import RequestDatasetCollection


def compute(dataset,calc,calc_grouping,tile_dimension,verbose=False,prefix=None):
    '''
    :type dataset: RequestDatasetCollection
    '''
    assert(isinstance(dataset,RequestDatasetCollection))
    assert(type(calc) in (list,tuple))
    
    tile_dimension = int(tile_dimension)
    if tile_dimension <= 0:
        raise(ValueError('"tile_dimension" must be greater than 0'))
    
    orig_oc = ocgis.env.OPTIMIZE_FOR_CALC
    ocgis.env.OPTIMIZE_FOR_CALC = False
    
    try:
        ## load some data into the optimize store
        print('loading into optimize store...')
        for rd in dataset:
            if verbose: print('request dataset',rd.alias)
            ocgis.env._optimize_store[rd.alias] = {}
            ocgis.env._optimize_store[rd.alias]['_value_datetime'] = rd.ds.temporal.value_datetime
            ocgis.env._optimize_store[rd.alias]['_bounds_datetime'] = rd.ds.temporal.bounds_datetime
            if calc_grouping is not None:
                rd.ds.temporal.set_grouping(calc_grouping)
                ocgis.env._optimize_store[rd.alias]['group'] = rd.ds.temporal.group
            rd._ds = None
        
        ## tell the software we are optimizing for calculations   
        ocgis.env.OPTIMIZE_FOR_CALC = True
        ods = NcDataset(request_dataset=dataset[0])
        shp = ods.spatial.grid.shape

        if verbose: print('getting schema...')
        schema = tile.get_tile_schema(shp[0],shp[1],tile_dimension)
        if verbose: print('getting fill file...')
        fill_file = ocgis.OcgOperations(dataset=dataset,file_only=True,
                                      calc=calc,calc_grouping=calc_grouping,
                                      output_format='nc',prefix=prefix).execute()
        if verbose: print('output file is: {0}'.format(fill_file))
        if verbose:
            lschema = len(schema)
            print('tile count: {0}'.format(lschema))
        fds = nc.Dataset(fill_file,'a')
        if verbose:
            progress = ProgressBar('tiles progress')
        for ctr,indices in enumerate(schema.itervalues(),start=1):
            row = indices['row']
            col = indices['col']
            ret = ocgis.OcgOperations(dataset=dataset,slice=[None,row,col],
                                calc=calc,calc_grouping=calc_grouping).execute()
            for vref,v in iter_variable_values(ret[1],fds):
                if len(vref.shape) == 3:
                    vref[:,row[0]:row[1],col[0]:col[1]] = v
                elif len(vref.shape) == 4:
                    vref[:,:,row[0]:row[1],col[0]:col[1]] = v
                else:
                    raise(NotImplementedError(vref.shape))
                fds.sync()
            if verbose:
                progress.progress(int((float(ctr)/lschema)*100))
                
        fds.close()
    finally:
        ocgis.env.OPTIMIZE_FOR_CALC = orig_oc
        ocgis.env._optimize_store = {}
    if verbose:
        progress.endProgress()
        print('complete.')
    return(fill_file)

def iter_variable_values(coll,fds):
    if type(coll) == CalcCollection:
        for variable in coll.variables.iterkeys():
            ref = coll.calc[variable]
            for k,v in ref.iteritems():
                vref = fds.variables[k]
                yield(vref,v)
    elif type(coll) == MultivariateCalcCollection:
        for calc_name,calc_value in coll.calc.iteritems():
            vref = fds.variables[calc_name]
            yield(vref,calc_value)
    else:
        raise(NotImplementedError(type(coll)))
