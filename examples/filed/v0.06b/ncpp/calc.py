import cProfile
import os
import tempfile
from subprocess import check_call

import ocgis

ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/bcca/obs/tasmax/1_8deg'


def single_year():
    rd = ocgis.RequestDataset(uri='gridded_obs.tasmax.OBS_125deg.daily.1999.nc',
                              variable='tasmax')
    calc = [
        {'func': 'freq_perc', 'name': 'perc_95', 'kwds': {'perc': 0.95, 'round_method': 'ceil'}},
        #            {'func':'mean','name':'mean'}
    ]
    calc_grouping = ['month', 'year']
    snippet = False
    select_ugid = None
    #    select_ugid = [32]
    geom = None
    #    geom = 'state_boundaries'
    ops = ocgis.OcgOperations(dataset=rd, snippet=snippet, geom=geom, select_ugid=select_ugid,
                              aggregate=False, spatial_operation='intersects',
                              output_format='nc', calc=calc, calc_grouping=calc_grouping)
    ret = ops.execute()
    return (ret)


def combine():
    ## file name template to insert year
    template = 'gridded_obs.tasmax.OBS_125deg.daily.{year}.nc'
    years = [1950, 1999]  # range of years
    ## output filename 
    outname = 'gridded_obs.tasmax.OBS_125deg.daily.{0}-{1}.nc'.format(*years)
    ## path to output file
    outfile = os.path.join(tempfile.gettempdir(), outname)
    ## make the file if it does not exist
    if not os.path.exists(outfile):
        ## make list of files to concatenate
        cfiles = []
        for year in range(years[0], years[1] + 1):
            cfiles.append(os.path.join(ocgis.env.DIR_DATA, template.format(year=year)))
        ## combine the files
        sargs = ['ncrcat'] + cfiles + [outfile]
        print('combining files...')
        check_call(sargs)
        print('files combined.')
    else:
        print('file already combined.')


def decade():
    ## file name template to insert year
    template = 'gridded_obs.tasmax.OBS_125deg.daily.{year}.nc'
    years = [1990, 1999]  # range of years
    ## output filename 
    outname = 'gridded_obs.tasmax.OBS_125deg.daily.{0}-{1}.nc'.format(*years)
    ## path to output file
    outfile = os.path.join(tempfile.gettempdir(), outname)
    ## make the file if it does not exist
    if not os.path.exists(outfile):
        ## make list of files to concatenate
        cfiles = []
        for year in range(years[0], years[1] + 1):
            cfiles.append(os.path.join(ocgis.env.DIR_DATA, template.format(year=year)))
        ## combine the files
        sargs = ['ncrcat'] + cfiles + [outfile]
        print('combining files...')
        check_call(sargs)
        print('files combined.')
    else:
        print('file already combined.')

    ## build ocgis operations ##################################################

    rd = ocgis.RequestDataset(uri=outfile, variable='tasmax')
    calc = [
        #            {'func':'freq_perc','name':'perc_99','kwds':{'perc':0.99,'round_method':'ceil'}},
        #            {'func':'freq_perc','name':'perc_95','kwds':{'perc':0.95,'round_method':'ceil'}},
        #            {'func':'freq_perc','name':'perc_90','kwds':{'perc':0.90,'round_method':'ceil'}},
        {'func': 'mean', 'name': 'mean'}]
    for c in calc:
        print(write_calc(rd, [c]))


def write_calc(rd, calc):
    ocgis.env.OVERWRITE = True

    calc_grouping = ['month']
    snippet = False
    prefix = 'calc_{0}'.format(calc[0]['name'])
    select_ugid = None
    #    select_ugid = [32]
    geom = None
    #    geom = 'state_boundaries'
    ops = ocgis.OcgOperations(dataset=rd, snippet=snippet, geom=geom, select_ugid=select_ugid,
                              aggregate=False, spatial_operation='intersects',
                              output_format='nc', calc=calc, calc_grouping=calc_grouping,
                              prefix=prefix)
    ret = ops.execute()
    return (ret)


def profile():
    cProfile.run('decade()', 'profile.log')


if __name__ == '__main__':
    #    ret = single_year()
    #    ret = decade()
    #    profile()
    combine()
