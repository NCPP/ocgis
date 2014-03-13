from subprocess import check_call
import logging as log


DOWNLOAD = '/usr/local/climate_data/maurer/bcca/obs/tasmax/1_8deg'
YEARS = [1950,1999]
BASE_URL = 'ftp://gdo-dcp.ucllnl.org/pub/dcp/archive/bcca/obs/tasmax/1_8deg/{0}'
FN_TEMPLATE = 'gridded_obs.tasmax.OBS_125deg.daily.{year}.nc'


def main():
    
    log.basicConfig(level=log.INFO,
                format='%(asctime)s %(levelname)s %(message)s',
                filename='ncpp.log',
                filemode='w')
    
    files = []
    for year in range(YEARS[0],YEARS[1]+1):
        files.append(FN_TEMPLATE.format(year=year))
        
    for f in files:
        url = BASE_URL.format(f)
        check_call(['wget','-P',DOWNLOAD,url])
        log.info('downloaded: {0}'.format(url))
 

if __name__ == '__main__':
    main()