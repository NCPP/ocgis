import ocgis
from ocgis.calc import tile
import netCDF4 as nc
from ocgis.util.helpers import ProgressBar
from ocgis.api.request.base import RequestDatasetCollection
import numpy as np


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
        
        ## tell the software we are optimizing for calculations   
        ocgis.env.OPTIMIZE_FOR_CALC = True
        ods = dataset[0].get()
#        ods = NcDataset(request_dataset=dataset[0])
        shp = ods.shape[-2:]

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
            ret = ocgis.OcgOperations(dataset=dataset,slice=[None,None,None,row,col],
                                calc=calc,calc_grouping=calc_grouping).execute()
            for field_map in ret.itervalues():
                for field in field_map.itervalues():
                    for alias,variable in field.variables.iteritems():
                        vref = fds.variables[alias]
                        if len(vref.shape) == 3:
                            vref[:,row[0]:row[1],col[0]:col[1]] = np.squeeze(variable.value)
                        elif len(vref.shape) == 4:
                            vref[:,:,row[0]:row[1],col[0]:col[1]] = np.squeeze(variable.value)
                        else:
                            raise(NotImplementedError(vref.shape))
                        fds.sync()
#                        import ipdb;ipdb.set_trace()
#            for vref,v in iter_variable_values(ret[1],fds):
#                if len(vref.shape) == 3:
#                    vref[:,row[0]:row[1],col[0]:col[1]] = v
#                elif len(vref.shape) == 4:
#                    vref[:,:,row[0]:row[1],col[0]:col[1]] = v
#                else:
#                    raise(NotImplementedError(vref.shape))
#                fds.sync()
            if verbose:
                progress.progress(int((float(ctr)/lschema)*100))
                
        fds.close()
    finally:
        ocgis.env.OPTIMIZE_FOR_CALC = orig_oc
    if verbose:
        progress.endProgress()
        print('complete.')
    return(fill_file)

#def iter_variable_values(coll,fds):
#    if type(coll) == CalcCollection:
#        for variable in coll.variables.iterkeys():
#            ref = coll.calc[variable]
#            for k,v in ref.iteritems():
#                vref = fds.variables[k]
#                yield(vref,v)
#    elif type(coll) == MultivariateCalcCollection:
#        for calc_name,calc_value in coll.calc.iteritems():
#            vref = fds.variables[calc_name]
#            yield(vref,calc_value)
#    else:
#        raise(NotImplementedError(type(coll)))
