import ocgis
from ocgis.calc import tile
import netCDF4 as nc
from ocgis.util.helpers import ProgressBar
from ocgis.interface.nc.dataset import NcDataset


def compute(dataset,calc,calc_grouping,tile_dimension,verbose=False,prefix=None):
    tile_dimension = int(tile_dimension)
    if tile_dimension <= 0:
        raise(ValueError('"tile_dimension" must be greater than 0'))
    
    orig_oc = ocgis.env.OPTIMIZE_FOR_CALC
    ocgis.env.OPTIMIZE_FOR_CALC = True
    try:
        ods = NcDataset(request_dataset=dataset)
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
            for variable in ret[1].variables.iterkeys():
                ref = ret[1].calc[variable]
                for k,v in ref.iteritems():
                    vref = fds.variables[k]
                    if len(vref.shape) == 3:
                        vref[:,row[0]:row[1],col[0]:col[1]] = v
                    elif len(vref.shape) == 4:
                        vref[:,:,row[0]:row[1],col[0]:col[1]] = v
                    else:
                        raise(NotImplementedError)
                    fds.sync()
            if verbose:
                progress.progress(int((float(ctr)/lschema)*100))
        fds.close()
    finally:
        ocgis.env.OPTIMIZE_FOR_CALC = orig_oc
    if verbose:
        progress.endProgress()
        print('complete.')
    return(fill_file)
