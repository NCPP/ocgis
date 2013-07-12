import ocgis
import os
import numpy as np
from ocgis.interface.projection import PolarStereographic


### lower logging level to debug output
#ocgis.env.DEBUG = False
### print more information to the terminal
#ocgis.env.VERBOSE = False
### set snippet to false to return all data
#snippet = False
### output directory
#ocgis.env.DIR_OUTPUT = '/tmp/narccap'
### the directory containing the target data
#ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
### push data to a common reference projection
#ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True


def parse_narccap_filenames(folder):
    ## parse data directory into requeset datasets
    rds = []
    filenames = os.listdir(folder)
    pieces = np.empty((len(filenames),4),dtype=object)
    ## split filenames into parts and collect those into groups.
    for ii,filename in enumerate(filenames):
        pieces[ii,0:3] = filename.split('_')[:-1]
        pieces[ii,-1] = filename
    for variable in np.unique(pieces[:,0]).flat:
        for gcm in np.unique(pieces[:,1]).flat:
            for rcm in np.unique(pieces[:,2]).flat:
                idx = np.all(pieces[:,0:3] == [variable,gcm,rcm],axis=1)
                if not idx.any():
                    continue
                ## there are multiple filepaths for each request dataset taking
                ## advantage of time dimension concatenation.
                uris = pieces[idx,-1].tolist()
                alias = variable+'_'+gcm+'_'+rcm
                ## this gcm-rcm combination does not have false_easting and
                ## false_northing attributes correctly filled.
                if gcm == 'ECP2' and rcm == 'ncep':
                    s_proj = PolarStereographic(60.0,90.0,263.0,4700000.0,8400000.0)
                else:
                    s_proj = None
                rd = ocgis.RequestDataset(uri=uris,alias=alias,
                            variable=variable,s_proj=s_proj,
                            meta={'gcm':gcm,'rcm':rcm})
                rds.append(rd)
    return(rds)

### get the request datasets from the filenames
#dataset = parse_narccap_filenames(ocgis.env.DIR_DATA)
### parameterize the calculations
#calc = [{'func':'mean','name':'mean'},
#        {'func':'median','name':'median'},
#        {'func':'max','name':'max'},
#        {'func':'min','name':'min'}]
### groups the time variable by month and year
#calc_grouping = ['month','year']
### make the operations object. the data will be written to csv+ format (includes
### shapefile overviews of geometries), subsetted by the CO watershed with unique
### geometry identifier of 5, and aggregated to the watershed geoemtry. the geomtric
### abstraction is "point" because the timeslice experiments have bounds.
#ops = ocgis.OcgOperations(dataset=dataset,calc=calc,calc_grouping=calc_grouping,
#                          output_format='csv+',geom='co_watersheds',aggregate=True,
#                          abstraction='point',select_ugid=[5],snippet=snippet)
### return the path to the ouput csv file.
#ret = ops.execute()
#print(ret)
