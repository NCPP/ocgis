import os
import numpy as np
from ocgis.interface.projection import PolarStereographic
import ocgis


def parse_narccap_filenames(folder):
    '''
    :param folder: path to the folder containing NARCCAP datasets
    :type folder: str
    :returns: sequence of RequestDataset objects
    :rtype: sequence
    '''
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
                ## sort the URIs to ensure proper ordering.
                uris.sort()
                ## this is the alias which is unique to a dataset.
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