import ocgis
import os
import numpy as np
from ocgis.interface.projection import PolarStereographic


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
            if gcm == 'ECP2' and rcm == 'ncep':
                s_proj = PolarStereographic(60.0,90.0,263.0,4700000.0,8400000.0)
            else:
                s_proj = None
            rd = ocgis.RequestDataset(uri=uris,alias=alias,
                        variable=variable,s_proj=s_proj,
                        meta={'gcm':gcm,'rcm':rcm})
            rds.append(rd)

#rds = [rds[0]]
#import ipdb;ipdb.set_trace()

# write overview shapefiles
#done = []
#for rd in rds:
#    if rd.variable == 'pr' and rd.meta not in done:
#        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='shp',
#                                  prefix=rd.meta['gcm']+'_'+rd.meta['rcm'])
#        ops.execute()
#        done.append(rd.meta)

## these are the calculations to perform
calc = [{'func':'mean','name':'mean'},
        {'func':'median','name':'median'},
        {'func':'max','name':'max'},
        {'func':'min','name':'min'}]
#calc = None
calc_grouping = ['month','year']

## the operations for index calculation
ops = ocgis.OcgOperations(dataset=rds,snippet=False,calc=calc,calc_grouping=calc_grouping,
                          output_format='csv+',geom=geom,abstraction='point')
ret = ops.execute()

import ipdb;ipdb.set_trace()