import ocgis
import os
import netCDF4 as nc
from ocgis.interface.projection import get_projection, WGS84
import sys; sys.path.append('/home/local/WX/ben.koziol/Dropbox/UsefulScripts/python')
from report import Report
from helpers import parse_narccap_filenames
from tempfile import NamedTemporaryFile
import csv
from ocgis.interface.nc.dataset import NcDataset
import tempfile
import logging


def generate_projection_report(folder):
    '''
    Generate a projection report for the NARCCAP data.
    
    :param folder: Path to the folder containing the NARCCAP example data.
    :type folder: str
    :returns: Path to the report file.
    :rtype: str
    '''
    report = Report()
    files = os.listdir(folder)
    files = filter(lambda x: x.endswith('.nc'),files)
    files = map(lambda x: os.path.join(folder,x),files)
    for f in files:
        report.add('Filename: {0}'.format(os.path.split(f)[1]),add_break=False)
        rootgrp = nc.Dataset(f,'r')
        try:
            projection = get_projection(rootgrp)
            report.add('OpenClimateGIS Projection Class: {0}'.format(projection.__class__))
            report.add('OpenClimateGIS Computed PROJ4 String: {0}'.format(projection.proj4_str))
            try:
                ref_var = rootgrp.variables[projection.variable_name]
                report.add('NetCDF Variable/Attribute Dump:')
                report.add(projection.variable_name,indent=1)
                for attr in ref_var.ncattrs():
                    report.add('{0}: {1}'.format(attr,ref_var.getncattr(attr)),indent=2)
            except KeyError:
                ## likely WGS84
                if isinstance(projection,WGS84):
                    continue
                else: raise
            finally:
                report.add_break()
                report.add('-'*89)
                report.add_break()
        finally:
            rootgrp.close()
    report.write('/tmp/foo.txt')
    import webbrowser;webbrowser.open('/tmp/foo.txt')

def generate_dataset_report(folder):
    '''
    Generate a descriptive report on the data included in the use case.
    
    :param folder: Path to the folder containing the NARCCAP example data.
    :type folder: str
    :returns: Path to the report file.
    :rtype: str
    '''
    
    ocgis.env.DIR_DATA = folder
    rds = parse_narccap_filenames(folder)
    rdc = ocgis.RequestDatasetCollection(rds)
    headers = ['DID','Filenames','Variable','Time Start','Time End']
    (fd,name) = tempfile.mkstemp(suffix='.csv')
    f = open(name,'w')
    writer = csv.writer(f)
    writer.writerow(headers)
    for rd in rdc:
        logging.info(rd)
        ds = NcDataset(rd)
        to_write = [rd.did,
                    [os.path.split(uri)[1] for uri in rd.uri],
                    rd.variable,
                    ds.temporal.value[0],
                    ds.temporal.value[-1]]
        writer.writerow(to_write)
    f.close()
    return(name)
            
if __name__ == '__main__':
    folder = '/usr/local/climate_data/narccap'
    
#    generate_projection_report(folder)
    ret = generate_dataset_report(folder)
    import ipdb;ipdb.set_trace()