from __future__ import division

import os

import numpy as np
from mpi4py import MPI

import ocgis
from ocgis.exc import ExtentError

# Code adopted from Ben Koziol and written by Joshua Sims and Laura Briley,
# December 2013 for GLISA
# this code uses openclimate gis to subset a dataset based on lat/lon range
# and automatically splits the different models between differnent threads using MPI
# (Message Passing Interface)
# run command: mpirun -np [number cores] python multithread.py

####setup MPI world######
comm = MPI.COMM_WORLD
### comm.size = number of threads
### comm.rank = thread number

model = 'giss_aom'
variable_name = 'tmin'
variable_type = 'temp'
# directory where the climate data sits
ocgis.env.DIR_DATA = '/Users/joshsims/gcModels/wicci_data/' + model
# directory where you want to write data
ocgis.env.DIR_OUTPUT = '/Users/joshsims/gcModels/wicci_climdiv/' + model
# directory ABOVE shapefile directory
ocgis.env.DIR_GEOMCABINET = '/Users/joshsims/gcModels/'
ocgis.env.OVERWRITE = True
ocgis.env.DEBUG = True
files = os.listdir(ocgis.env.DIR_DATA)  # get all files in data directory
ncfiles = []

# climate division list
cdList = ['1101', '1102', '1103', '1104', '1105', '1106', '1107', '1108', \
          '1201', '1202', '1203', '1204', '1205', '1206', '1207', '1208', '1209', \
          '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', \
          '2101', '2102', '2103', '2104', '2105', '2106', '2107', '2108', '2109', \
          '3001', '3002', '3003', '3004', '3005', '3006', '3007', '3008', '3009', \
          '3301', '3302', '3303', '3304', '3305', '3306', '3307', '3308', '3309', \
          '3601', '3602', '3603', '3604', '3605', '3606', '3607', '3608', '3609', '3610', \
          '4701', '4702', '4703', '4704', '4705', '4706', '4707', '4708', '4709']

print('collecting files...')

# go through all files and only include those that have the correct variable
for f in range(len(files)):
    filename = files[f]
    x = filename.find(variable_type)
    # if the filename contains the variable name
    if x != -1:
        # append the filename to the list of files to operate
        ncfiles.append(filename)

print ncfiles

###############set up arrays for MPI Scatterv####################
# big letter variable names represent totals
# my_ represents the array each thread will process

# number of total items to process
N = len(ncfiles)

# number of items to send to each thread (before remainder)
my_N = N // (comm.size)

# remainder
r_N = N % (comm.size)
# print my_N

# create array of items to scatter
if comm.rank == 0:
    A = np.arange(N, dtype=int)
#   print A
else:
    A = None

# create arrays to catch the scatter, attach remainders
if comm.rank <= (r_N - 1):
    my_A = np.zeros(my_N + 1, dtype=int)
else:
    my_A = np.zeros(my_N, dtype=int)

# set up sendcounts, number of items to send each array
sendcounts = ()
send_mult = 2
print send_mult
for x in range(r_N):
    sendcounts = sendcounts + ((my_N + 1) * send_mult,)
for y in range(comm.size - r_N):
    sendcounts = sendcounts + (my_N * send_mult,)

# set up displacement counts
displace = ()
dis = 0
for d in range(r_N + 1):
    displace = displace + (dis,)
    if r_N != 0 and len(displace) <= (r_N):
        dis += ((my_N + 1) * 2)
    elif len(displace) <= (r_N):
        dis += ((my_N + 1) * 2)
    else:
        dis += (my_N * 2)
for e in range(comm.size - (r_N + 1)):
    displace = displace + (dis,)
    dis += (my_N * 2)
if comm.rank == 0:
    print sendcounts
    print displace

# Scatter data into my_A arrays
comm.Scatterv([A, sendcounts, displace, MPI.INT], my_A)

print(my_A)

#########################start processing here ##################################

for i in my_A:
    # request data set
    rds = [ocgis.RequestDataset(ncfiles[i], variable_name)]
    print('operating on '), ncfiles[i]
    # get the file name (minus the '.nc'
    # to later pass to the folder prefix of where the data are saved
    filename = ncfiles[i]
    savename = filename[0:len(filename) - 3]
    # define the geometry
    geom = 'climate_divisions'
    # define the file output format
    output_format = 'csv'
    # if True, only perform operation on first time increment
    snippet = False

    # climate division ugid (identifyers) for MN,WI,IL,IN,MI,OH,PA,NY
    for select_ugid in cdList:
        # save the output file as the division ID + Filename
        prefix = str(select_ugid) + str('_') + str(savename)
        print 'working on division ', prefix
        try:
            ops_agg = ocgis.OcgOperations(dataset=rds, aggregate=True, geom=geom, \
                                          select_ugid=[select_ugid], spatial_operation='clip', snippet=snippet, \
                                          output_format=output_format, prefix=prefix, interpolate_spatial_bounds=True)

            print('executing aggregation...')
            path_agg = ops_agg.execute()
        # catch extent errors
        except ExtentError:
            print 'error on ', select_ugid

print('success.')
