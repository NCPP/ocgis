import ocgis
import os
import numpy as np


ocgis.env.VERBOSE = True


## set snippet to false to return all data
snippet = False
## city center coordinate
geom = [-97.74278,30.26694]
## output directory
ocgis.env.DIR_OUTPUT = '/tmp/narccap'
## the directory containing the target data
#ocgis.env.DIR_DATA = '/media/Helium Backup/narccap'
ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
## push data to a common reference projection
ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True


rds = []
filenames = os.listdir(ocgis.env.DIR_DATA)
pieces = np.empty((len(filenames),4),dtype=object)
for ii,filename in enumerate(filenames):
    pieces[ii,0:3] = filename.split('_')[:-1]
    pieces[ii,-1] = filename
for variable in np.unique(pieces[:,0]).flat:
    for gcm in np.unique(pieces[:,1]).flat:
        for rcm in np.unique(pieces[:,2]).flat:
            idx = np.all(pieces[:,0:3] == [variable,gcm,rcm],axis=1)
            if not idx.any():
                continue
            uris = pieces[idx,-1].tolist()
            alias = variable+'_'+gcm+'_'+rcm
            rds.append(ocgis.RequestDataset(uri=uris,alias=alias,variable=variable))

rds = rds[-3:]
#import ipdb;ipdb.set_trace()

#import ipdb;ipdb.set_trace()
### construct aliases for the datasets
#aliases = [fn.split('_')[1] for fn in filenames]
### files all use the same variable
#variable = 'pr'
### make the request datasets
#rds = [ocgis.RequestDataset(uri=fn,variable=variable,alias=alias) for fn,alias in zip(filenames,aliases)]

# write overview shapefiles
#for rd in rds:
#    ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='shp',prefix=rd.alias)
#    ops.execute()

## these are the calculations to perform
calc = [{'func':'mean','name':'mean'}]
#calc = None
calc_grouping = ['month','year']

## the operations for index calculation
ops = ocgis.OcgOperations(dataset=rds,snippet=snippet,calc=calc,calc_grouping=calc_grouping,
                          output_format='shp',geom=geom)
ret = ops.execute()
import ipdb;ipdb.set_trace()