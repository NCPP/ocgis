import unittest
from shapely.geometry.polygon import Polygon
from util.ncwrite import NcSpatial, NcTime, NcVariable, NcWrite, NcSubset
import datetime, re
import os
import numpy as np
from netCDF4 import Dataset
import in_memory_oo_multi_core as ncconv


class TestSimpleNc(unittest.TestCase):

    def get_uri(self,bounds=Polygon(((0,0),(10,0),(10,15),(0,15))),rng=[datetime.datetime(2007,10,1),datetime.datetime(2007,10,3)],res=5,constant=5,nlevels=1,path=None,bnds=True,seed=None):
        """
        constant=5 -- provides a constant value when generating data to fill
            a NC variable. set to None to generate random values.
        nlevels=1 -- a value greater than 1 will create a NC with a fourth level
            dimension. the input values determines the number of levels.
        """
        ncspatial = NcSpatial(bounds,res,add_bounds=bnds)
        
        interval = datetime.timedelta(days=1)
        nctime = NcTime(rng,interval)
        
        ncvariable = NcVariable("Prcp","mm",constant=constant,seed=seed)
        
        ncw = NcWrite(ncvariable,ncspatial,nctime,nlevels=nlevels)
        uri = ncw.write(path)
        return(uri)

    def test_get_uri(self):
        uri = self.get_uri(bnds=False)
        self.assertTrue(os.path.exists(uri))
        d = Dataset(uri,'r')
        sh = d.variables["Prcp"].shape
        self.assertEqual(sh,(3,3,2))
        d.close()



    def _getCtrd(self,element):
        g = element['geometry']
        return (g.centroid.x,g.centroid.y)


    def setUp(self):
        self.singleLayer = self.get_uri(bounds=Polygon(((0,0),(40,0),(40,20),(0,40))),rng=[datetime.datetime(2000,1,1),datetime.datetime(2000,1,10)],res=10,constant=None,seed=1)
        self.multiLayer = self.get_uri(bounds=Polygon(((0,0),(40,0),(40,20),(0,40))),rng=[datetime.datetime(2000,1,1),datetime.datetime(2000,1,10)],res=10,constant=None,seed=1,nlevels=4)

    def _access(self,uri,polygons,temporal,dissolve,clip,levels,subdivide,subres):
        POLYINT = polygons
        
        dataset = uri

        TEMPORAL = temporal
        DISSOLVE = dissolve
        CLIP = clip
        VAR = 'Prcp'
        kwds = {
            'rowbnds_name': 'bounds_latitude', 
            'colbnds_name': 'bounds_longitude',
            'time_units': 'days since 1800-01-01 00:00:00 0:00',
            'level_name': 'level',
            'calendar': 'gregorian'
        }
        ## open the dataset for reading
        ## make iterable if only a single polygon requested
        if type(POLYINT) not in (list,tuple): POLYINT = [POLYINT]
        ## convenience function for multiple polygons
        elements = ncconv.multipolygon_multicore_operation(dataset,
                                        VAR,
                                        POLYINT,
                                        time_range=TEMPORAL,
                                        clip=CLIP,
                                        dissolve=DISSOLVE,
                                        levels = levels,
                                        ocgOpts=kwds,
                                        subdivide=subdivide,
                                        subres = subres
                                        )

        return elements

    def testSingleLayerSubset(self):
        'Subset a netCDF4 file with a single layer of data'

        print 'single layer subset'
        kwds = {
            'time_units': 'days since 1950-1-1 0:0:0.0',
        }
        dataset = "./bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc"
        ocg = ncconv.OcgDataset(dataset,**kwds)

        #polygon = Polygon(((-90,40),(-80,40),(-80,50),(-90,50)))
        polygon = Polygon(((-90,40),(-80,40),(-80,42),(-90,50)))
        time_range = [datetime.datetime(1950,2,1),datetime.datetime(1950,5,1)]
        NcSubset("test.nc",ocg,"Prcp",polygon,time_range)


        elements = ncconv.multipolygon_multicore_operation(dataset,
                                        'Prcp',
                                        [polygon],
                                        time_range=time_range,
                                        clip=False,
                                        dissolve=False,
                                        levels = None,
                                        ocgOpts=kwds,
                                        subdivide=True,
                                        )

        elements2 = ncconv.multipolygon_multicore_operation('test.nc',
                                        'Prcp',
                                        [None],
                                        time_range=time_range,
                                        clip=False,
                                        dissolve=False,
                                        levels = None,
                                        ocgOpts=kwds,
                                        subdivide=True,
                                        )

        self.assertEqual(len(elements),len(elements2))

        l1 = [list(x['geometry'].exterior.coords) for x in elements]

        l2 = [list(x['geometry'].exterior.coords) for x in elements2]

        l1.sort()
        l2.sort()
        np.testing.assert_array_almost_equal(l1,l2)

        l1 = [x['properties']['Prcp'] for x in elements]

        l2 = [x['properties']['Prcp'] for x in elements2]

        l1.sort()
        l2.sort()
        self.assertEqual(l1,l2)

        l1 = [x['properties']['timestamp'] for x in elements]

        l2 = [x['properties']['timestamp'] for x in elements2]

        l1.sort()
        l2.sort()
        self.assertEqual(l1,l2)

    def testMultiLayerSubset(self):
        'Subset a netCDF4 file with multiple layers of data'

        print 'multi layer subset'

        kwds = {
            'rowbnds_name': 'lat_bnds', 
            'colbnds_name': 'lon_bnds',
            'time_units': 'days since 1800-1-1 00:00:0.0',
            'level_name': 'lev'
        }
        dataset = './pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
        ocg = ncconv.OcgDataset(dataset,**kwds)
        levels = [0,1,2,3]

        polygon = Polygon(((0,0),(0,5),(20,20),(20,0)))
        #time_range = [datetime.datetime(1960,3,16),datetime.datetime(1961,5,16)]
        time_range = [datetime.datetime(1960,3,16),datetime.datetime(1961,3,16)]
        NcSubset("test2.nc",ocg,"cl",polygon,time_range,Levels=levels,lat_name='lat',lon_name='lon')


        elements = ncconv.multipolygon_multicore_operation(dataset,
                                        'cl',
                                        [polygon],
                                        time_range=time_range,
                                        clip=False,
                                        dissolve=False,
                                        levels = levels,
                                        ocgOpts=kwds,
                                        subdivide=False,
                                        )

        elements2 = ncconv.multipolygon_multicore_operation('test2.nc',
                                        'cl',
                                        [None],
                                        time_range=time_range,
                                        clip=False,
                                        dissolve=False,
                                        levels = levels,
                                        ocgOpts=kwds,
                                        subdivide=False,
                                        )

        print len(elements),len(elements2)
        self.assertEqual(len(elements),len(elements2))

        l1 = [list(x['geometry'].exterior.coords) for x in elements]

        l2 = [list(x['geometry'].exterior.coords) for x in elements2]

        l1.sort()
        l2.sort()
        np.testing.assert_array_almost_equal(l1,l2)

        l1 = [x['properties']['cl'] for x in elements]

        l2 = [x['properties']['cl'] for x in elements2]

        l1.sort()
        l2.sort()
        self.assertEqual(l1,l2)

        l1 = [x['properties']['timestamp'] for x in elements]

        l2 = [x['properties']['timestamp'] for x in elements2]

        l1.sort()
        l2.sort()
        self.assertEqual(l1,l2)

        


#-------------------------------------output-----------------------------

    def test_tabular(self):
        "Tabular output format"

        layer = self.singleLayer
        poly = Polygon(((0,0),(20,0),(20,20),(0,20)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        
        ncconv.as_tabular(elements,'Prcp',path='./test_tabular.txt')
        tf = open('./test_tabular.txt','r')

        E = 1000000000000.0

        lines = [line.replace('\n','').split(',') for line in tf][1:]
        self.assertEqual(lines[0][:2],["1","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[0][2]),1.6243454217910767,6)
        self.assertAlmostEqual(float(lines[0][3])/E,1220762936068.626/E,12)#1220762936068.626)

        self.assertEqual(lines[1][:2],["2","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[1][2]),0.61175638437271118,6)
        self.assertAlmostEqual(float(lines[1][3])/E,1220762936068.6296/E,12)

        self.assertEqual(lines[2][:2],["3","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[2][2]),0.86540764570236206,6)
        self.assertAlmostEqual(float(lines[2][3])/E,1184603064536.2717/E,12)

        self.assertEqual(lines[3][:2],["4","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[3][2]),2.3015387058258057,6)
        self.assertAlmostEqual(float(lines[3][3])/E,1184603064536.2744/E,12)
        #self.assertEqual(lines[0].replace('\n',''),"1,2000-01-01 00:00:00,1.6243454217910767,1220762936068.626")
        #self.assertEqual(lines[1].replace('\n',''),"2,2000-01-01 00:00:00,0.61175638437271118,1220762936068.6296")
        #self.assertEqual(lines[2].replace('\n',''),"3,2000-01-01 00:00:00,0.86540764570236206,1184603064536.2717")
        #self.assertEqual(lines[3].replace('\n',''),"4,2000-01-01 00:00:00,2.3015387058258057,1184603064536.2744")

        tf.close()
        
        poly = Polygon(((0,0),(10,0),(10,10),(0,10)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        ncconv.as_tabular(elements,'Prcp',wkt=True,path='./test_tabular.txt')
        tf = open('./test_tabular.txt','r')

        lines = [line.replace('\n','').split(',') for line in tf][1:]
        self.assertEqual(lines[0][:2],["1","2000-01-01 00:00:00"])
        self.assertEqual(','.join(lines[0][4:]),"'POLYGON ((0.0000000000000000 0.0000000000000000, 0.0000000000000000 10.0000000000000000, 10.0000000000000000 10.0000000000000000, 10.0000000000000000 0.0000000000000000, 0.0000000000000000 0.0000000000000000))'")
        self.assertAlmostEqual(float(lines[0][2]),1.6243454217910767,6)
        self.assertAlmostEqual(float(lines[0][3])/E,1220762936068.626/E,12)
        #self.assertEqual(lines[0].replace('\n',''),"1,2000-01-01 00:00:00,1.6243454217910767,1220762936068.626,'POLYGON ((0.0000000000000000 0.0000000000000000, 0.0000000000000000 10.0000000000000000, 10.0000000000000000 10.0000000000000000, 10.0000000000000000 0.0000000000000000, 0.0000000000000000 0.0000000000000000))'")

        tf.close()
        layer = self.multiLayer
        levels = [1]

        poly = Polygon(((0,0),(20,0),(20,20),(0,20)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        
        ncconv.as_tabular(elements,'Prcp',path='./test_tabular.txt')
        tf = open('./test_tabular.txt','r')

        lines = [line.replace('\n','').split(',') for line in tf][1:]
        self.assertEqual(lines[0][:2],["1","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[0][2]),0.17242820560932159,6)
        self.assertAlmostEqual(float(lines[0][4])/E,1220762936068.626/E,12)
        self.assertEqual(lines[0][3],'2')

        self.assertEqual(lines[1][:2],["2","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[1][2]),0.87785840034484863,6)
        self.assertAlmostEqual(float(lines[1][4])/E,1220762936068.6296/E,12)
        self.assertEqual(lines[1][3],'2')

        self.assertEqual(lines[2][:2],["3","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[2][2]),1.1006191968917847,6)
        self.assertAlmostEqual(float(lines[2][4])/E,1184603064536.2717/E,12)
        self.assertEqual(lines[2][3],'2')

        self.assertEqual(lines[3][:2],["4","2000-01-01 00:00:00"])
        self.assertAlmostEqual(float(lines[3][2]),1.144723653793335,6)
        self.assertAlmostEqual(float(lines[3][4])/E,1184603064536.2744/E,12)
        self.assertEqual(lines[3][3],'2')
        #self.assertEqual(lines[0].replace('\n',''),"1,2000-01-01 00:00:00,0.17242820560932159,2,1220762936068.626")
        #self.assertEqual(lines[1].replace('\n',''),"2,2000-01-01 00:00:00,0.87785840034484863,2,1220762936068.6296")
        #self.assertEqual(lines[2].replace('\n',''),"3,2000-01-01 00:00:00,1.1006191968917847,2,1184603064536.2717")
        #self.assertEqual(lines[3].replace('\n',''),"4,2000-01-01 00:00:00,1.144723653793335,2,1184603064536.2744")
        

    def test_keyTabular(self):
        "Tabular output format with foreign keys for time and geometry"
        

        layer = self.multiLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = [0]
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        ncconv.as_keyTabular(elements,'Prcp',path='./test_keyTabular.txt')
        tft = open('./test_keyTabular_time.txt','r')
        tfg = open('./test_keyTabular_geometry.txt','r')
        tfd = open('./test_keyTabular_data.txt','r')

        E=1000000000000

        lines = [line for line in tft][1:]

        self.assertEqual(lines[0].replace('\n',''),'1,2000-01-01 00:00:00')

        lines = [line.replace('\n','').split(',') for line in tfg][1:]

        self.assertEqual(lines[0][0],'1')
        self.assertAlmostEqual(float(lines[0][1])/E,1220762936068.626/E)

        lines = [line for line in tfd][1:]

        self.assertEqual(lines[0].replace('\n',''),'0,1,1,1.6243454217910767,1')

        tft.close()
        tfg.close()
        tfd.close()

        ncconv.as_keyTabular(elements,'Prcp',wkt=True,path='./test_keyTabular.txt')
        tft = open('./test_keyTabular_time.txt','r')
        tfg = open('./test_keyTabular_geometry.txt','r')
        tfd = open('./test_keyTabular_data.txt','r')

        lines = [line for line in tft][1:]

        self.assertEqual(lines[0].replace('\n',''),'1,2000-01-01 00:00:00')

        #lines = [line for line in tfg]

        #self.assertEqual(lines[0].replace('\n',''),"1,1220762936068.626,POLYGON ((0.0000000000000000 0.0000000000000000, 0.0000000000000000 10.0000000000000000, 10.0000000000000000 10.0000000000000000, 10.0000000000000000 0.0000000000000000, 0.0000000000000000 0.0000000000000000))")

        lines = [line.replace('\n','').split(',') for line in tfg][1:]

        self.assertEqual(lines[0][0],'1')
        self.assertAlmostEqual(float(lines[0][1])/E,1220762936068.626/E)
        self.assertEqual(",".join(lines[0][2:]),"POLYGON ((0.0000000000000000 0.0000000000000000, 0.0000000000000000 10.0000000000000000, 10.0000000000000000 10.0000000000000000, 10.0000000000000000 0.0000000000000000, 0.0000000000000000 0.0000000000000000))")

        lines = [line for line in tfd][1:]

        self.assertEqual(lines[0].replace('\n',''),'0,1,1,1.6243454217910767,1')

        tft.close()
        tfg.close()
        tfd.close()


        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,2)]
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        ncconv.as_keyTabular(elements,'Prcp',path='./test_keyTabular.txt')
        tft = open('./test_keyTabular_time.txt','r')
        tfg = open('./test_keyTabular_geometry.txt','r')
        tfd = open('./test_keyTabular_data.txt','r')

        lines = [line for line in tft][1:]

        self.assertEqual(lines[1].replace('\n',''),'2,2000-01-01 00:00:00')
        self.assertEqual(lines[0].replace('\n',''),'1,2000-01-02 00:00:00')

        lines = [line.replace('\n','').split(',') for line in tfg][1:]

        self.assertEqual(lines[0][0],'1')
        self.assertAlmostEqual(float(lines[0][1])/E,1220762936068.626/E)

        lines = [line for line in tfd][1:]

        self.assertEqual(lines[0].replace('\n',''),'1,1,1,0.48851814866065979,1')
        self.assertEqual(lines[1].replace('\n',''),'0,2,1,1.6243454217910767,1')

        tft.close()
        tfg.close()
        tfd.close()


        poly = Polygon(((0,0),(20,0),(20,10),(0,10)))   
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        ncconv.as_keyTabular(elements,'Prcp',path='./test_keyTabular.txt')
        tft = open('./test_keyTabular_time.txt','r')
        tfg = open('./test_keyTabular_geometry.txt','r')
        tfd = open('./test_keyTabular_data.txt','r')

        lines = [line for line in tft][1:]

        self.assertEqual(lines[0].replace('\n',''),'1,2000-01-01 00:00:00')

        lines = [line.replace('\n','').split(',') for line in tfg][1:]

        self.assertEqual(lines[0][0],'1')
        self.assertAlmostEqual(float(lines[0][1])/E,1220762936068.626/E)
        self.assertAlmostEqual(float(lines[0][1])/E,1220762936068.6296/E)

        #self.assertEqual(lines[0].replace('\n',''),'1,1220762936068.626')
        #self.assertEqual(lines[1].replace('\n',''),'2,1220762936068.6296')

        lines = [line for line in tfd][1:]

        self.assertEqual(lines[0].replace('\n',''),'0,1,1,1.6243454217910767,1')
        self.assertEqual(lines[1].replace('\n',''),'1,1,2,0.61175638437271118,1')

        tft.close()
        tfg.close()
        tfd.close()



#----------------------------------time, layers-----------------------------

    def test_LSSlMdRcd(self):
        "Local file, Single core, Single layer, Multiple date, Single Rectangle, Clip=False, Dissolve=False"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,10)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),10)

        c = [x['properties']['timestamp'].strftime("%Y%m%d") for x in elements]
        d = [repr(x) for x in xrange(20000101,20000111)]
        self.assertEqual(c,d)

        time = [datetime.datetime(2000,1,5),datetime.datetime(2000,1,7)]
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),3)

        c = [x['properties']['timestamp'].strftime("%Y%m%d") for x in elements]
        d = [repr(x) for x in xrange(20000105,20000108)]
        self.assertEqual(c,d)

    def test_LSMlSdRcd(self):
        "Local file, Single core, Single layer, Multiple date, Single Rectangle, Clip=False, Dissolve=False"
        #pick one point in the upper left
        layer = self.multiLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = [0]
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)

        c = [round(x['properties']['Prcp'],6) for x in elements]
        self.assertTrue(round(1.62434536,6) in c)


        levels = [0,1,2,3]
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),4)

        c = [round(x['properties']['Prcp'],6) for x in elements]
        self.assertTrue(round(1.62434536,6) in c and round(.172428208,6) in c and round(.687172700,6) in c and round(.120158952,6) in c)

        levels = [1,3]
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),2)

        c = [round(x['properties']['Prcp'],6) for x in elements]
        self.assertTrue(round(.172428208,6) in c and round(.120158952,6) in c)

#----------------------------------single thread tests----------------------------

    def test_LSSlSdRcd(self):
        "Local file, Single core, Single layer, Single date, Single Rectangle, Clip=False, Dissolve=False"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(5,5))

        poly = Polygon(((0,0),(10,0),(0,10),(10,10),(0,0)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),0)

        #pick one point in the middle
        poly = Polygon(((10,10),(20,10),(20,20),(10,20)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),1)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((15,15) in c)
        
        #pick 2 elements
        poly = Polygon(((0,0),(20,0),(20,10),(0,10)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),2)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (15,5) in c)

        #pick the top row
        poly = Polygon(((0,0),(40,0),(40,10),(0,10)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (15,5) in c and (25,5 in c) and (35,5) in c)

        #pick the left row
        poly = Polygon(((0,0),(10,0),(10,40),(0,40)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (5,15) in c and (5,25 in c) and (5,35) in c)

        #pick the upper everything but right and bottom row
        poly = Polygon(((0,0),(20,0),(20,20),(0,20)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (15,5) in c and (15,15) in c and (5,15) in c)

        #pick everything
        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),16)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5, 5) in c and (15, 5) in c and (25, 5) in c and (35, 5) in c)
        self.assertTrue((5,15) in c and (15,15) in c and (25,15) in c and (35,15) in c)
        self.assertTrue((5,25) in c and (15,25) in c and (25,25) in c and (35,25) in c)
        self.assertTrue((5,35) in c and (15,35) in c and (25,35) in c and (35,35) in c)

    def test_LSSlSdIcd(self):
        "Local file, Single core, Single layer, Single date, Irregular Polygon, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),10)
        c = [self._getCtrd(x) for x in elements]

        self.assertTrue((5,5) in c and (15,15) in c and (25,25) in c and (35,35) in c)
        self.assertTrue((15,5) in c and (25,15) in c and (35,25) in c)
        self.assertTrue((5,15) in c and (15,25) in c and (25,35) in c)


    def test_LSSlSdRCd(self):
        "Local file, Single core, Single layer, Single date, Single Rectangle, Clip=True, Dissolve=False"
        #pick one point in the middle
        layer = self.singleLayer
        poly = Polygon(((10,10),(20,10),(20,20),(10,20)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'

        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        self.assertEqual(len(elements),1)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((15,15) in c)


        #move the area to cover the intersection of 4 points
        poly = Polygon(((5,5),(15,5),(15,15),(5,15)))
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((7.5, 7.5) in c and (12.5, 7.5) in c and (12.5, 12.5) in c and (7.5, 12.5) in c)

        #pick everything
        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        self.assertEqual(len(elements),16)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5, 5) in c and (15, 5) in c and (25, 5) in c and (35, 5) in c)
        self.assertTrue((5,15) in c and (15,15) in c and (25,15) in c and (35,15) in c)
        self.assertTrue((5,25) in c and (15,25) in c and (25,25) in c and (35,25) in c)
        self.assertTrue((5,35) in c and (15,35) in c and (25,35) in c and (35,35) in c)

        #reduce selection by 5 units
        poly = Polygon(((5,5),(35,5),(35,35),(5,35)))
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        #print elements
        self.assertEqual(len(elements),16)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((7.5, 7.5) in c and (15, 7.5) in c and (25, 7.5) in c and (32.5, 7.5) in c)
        self.assertTrue((7.5,15) in c and (15,15) in c and (25,15) in c and (32.5,15) in c)
        self.assertTrue((7.5,25) in c and (15,25) in c and (25,25) in c and (32.5,25) in c)
        self.assertTrue((7.5,32.5) in c and (15,32.5) in c and (25,32.5) in c and (32.5,32.5) in c)

    def test_LSSlSdICd(self):
        "Local file, Single core, Single layer, Single date, Irregular Polygon, Clip=True, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)

        self.assertEqual(len(elements),10)
        c = [(round(self._getCtrd(x)[0],6),round(self._getCtrd(x)[1],6)) for x in elements]

        self.assertTrue((5,5) in c and (15,15) in c and (25,25) in c and (35,35) in c)
        self.assertTrue((round(15-5/3.0,6),round(5+5/3.0,6)) in c and (round(25-5/3.0,6),round(15+5/3.0,6)) in c and (round(35-5/3.0,6),round(25+5/3.0,6)) in c)
        self.assertTrue((round(5+5/3.0,6),round(15-5/3.0,6)) in c and (round(15+5/3.0,6),round(25-5/3.0,6)) in c and (round(25+5/3.0,6),round(35-5/3.0,6)) in c)

    def test_LSSlSdRcD(self):
        "Local file, Single core, Single layer, Single date, Single Rectangle, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(5,5))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.62434536,6)


        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.033812345,6)

    def test_LSSlSdIcD(self):
        "Local file, Single core, Single layer, Single date, Irregular Polygon, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((2.5,2.5),(17.5,2.5),(17.5,17.5),(2.5,17.5))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(10,10))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.350762025,6)


        poly = Polygon(((2.5,2.5),(2.5,37.5),(37.5,37.5),(37.5,2.5)))
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.033812345,6)


        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.31531396,6)

    def test_LSSlSdRCD(self):
        "Local file, Single core, Single layer, Single date, Single Rectangle, Clip=True, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(5,5))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.62434536,6)

        #pick everything
        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.033812345,6)

    def test_LSSlSdICD(self):
        "Local file, Single core, Single layer, Single date, Irregular Polygon, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((2.5,2.5),(17.5,2.5),(17.5,17.5),(2.5,17.5))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(10,10))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.350762025,6)


        poly = Polygon(((5,5),(5,35),(35,35),(35,5)))
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.124389726,6)


        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        #print poly.envelope
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.402930205,6)

    def test_LSSlSdMpCD(self):
        "Local file, Single core, Single layer, Single date, Multiple Polygons, Clip=False, Dissolve=True"

        layer = self.singleLayer
        poly = [Polygon(((0,0),(10,0),(10,10),(0,10))), Polygon(((20,20),(30,20),(30,30),(20,30)))]
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),2)
        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (25,25) in c)

        poly = [Polygon(((0,0),(20,0),(20,20),(0,20))), Polygon(((0,10),(20,10),(20,30),(0,30)))]
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),2)
        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((10,10) in c and (10,20) in c)

#----------------------------------multi thread tests----------------------------

    def test_LMSlSdRcd(self):
        "Local file, Multi core, Single layer, Single date, Single Rectangle, Clip=False, Dissolve=False"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(5,5))

        #pick one point in the middle
        poly = Polygon(((10,10),(20,10),(20,20),(10,20)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),1)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((15,15) in c)
        
        #pick 2 elements
        poly = Polygon(((0,0),(20,0),(20,10),(0,10)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),2)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (15,5) in c)

        #pick the top row
        poly = Polygon(((0,0),(40,0),(40,10),(0,10)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (15,5) in c and (25,5 in c) and (35,5) in c)

        #pick the left row
        poly = Polygon(((0,0),(10,0),(10,40),(0,40)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (5,15) in c and (5,25 in c) and (5,35) in c)

        #pick the upper everything but right and bottom row
        poly = Polygon(((0,0),(20,0),(20,20),(0,20)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (15,5) in c and (15,15) in c and (5,15) in c)

        #pick everything
        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)
        self.assertEqual(len(elements),16)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5, 5) in c and (15, 5) in c and (25, 5) in c and (35, 5) in c)
        self.assertTrue((5,15) in c and (15,15) in c and (25,15) in c and (35,15) in c)
        self.assertTrue((5,25) in c and (15,25) in c and (25,25) in c and (35,25) in c)
        self.assertTrue((5,35) in c and (15,35) in c and (25,35) in c and (35,35) in c)

    def test_LMSlSdIcd(self):
        "Local file, Multi core, Single layer, Single date, Irregular Polygon, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),10)
        c = [self._getCtrd(x) for x in elements]

        self.assertTrue((5,5) in c and (15,15) in c and (25,25) in c and (35,35) in c)
        self.assertTrue((15,5) in c and (25,15) in c and (35,25) in c)
        self.assertTrue((5,15) in c and (15,25) in c and (25,35) in c)


    def test_LMSlSdRCd(self):
        "Local file, Multi core, Single layer, Single date, Single Rectangle, Clip=True, Dissolve=False"
        #pick one point in the middle
        layer = self.singleLayer
        poly = Polygon(((10,10),(20,10),(20,20),(10,20)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'

        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        self.assertEqual(len(elements),1)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((15,15) in c)


        #move the area to cover the intersection of 4 points
        poly = Polygon(((5,5),(15,5),(15,15),(5,15)))
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        self.assertEqual(len(elements),4)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((7.5, 7.5) in c and (12.5, 7.5) in c and (12.5, 12.5) in c and (7.5, 12.5) in c)

        #pick everything
        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        self.assertEqual(len(elements),16)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5, 5) in c and (15, 5) in c and (25, 5) in c and (35, 5) in c)
        self.assertTrue((5,15) in c and (15,15) in c and (25,15) in c and (35,15) in c)
        self.assertTrue((5,25) in c and (15,25) in c and (25,25) in c and (35,25) in c)
        self.assertTrue((5,35) in c and (15,35) in c and (25,35) in c and (35,35) in c)

        #reduce selection by 5 units
        poly = Polygon(((5,5),(35,5),(35,35),(5,35)))
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)
        #print elements
        self.assertEqual(len(elements),16)

        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((7.5, 7.5) in c and (15, 7.5) in c and (25, 7.5) in c and (32.5, 7.5) in c)
        self.assertTrue((7.5,15) in c and (15,15) in c and (25,15) in c and (32.5,15) in c)
        self.assertTrue((7.5,25) in c and (15,25) in c and (25,25) in c and (32.5,25) in c)
        self.assertTrue((7.5,32.5) in c and (15,32.5) in c and (25,32.5) in c and (32.5,32.5) in c)

    def test_LMSlSdIcd(self):
        "Local file, Multi core, Single layer, Single date, Irregular Polygon, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,False,True,levels,subdivide,subres)

        self.assertEqual(len(elements),10)
        c = [(round(self._getCtrd(x)[0],6),round(self._getCtrd(x)[1],6)) for x in elements]

        self.assertTrue((5,5) in c and (15,15) in c and (25,25) in c and (35,35) in c)
        self.assertTrue((round(15-5/3.0,6),round(5+5/3.0,6)) in c and (round(25-5/3.0,6),round(15+5/3.0,6)) in c and (round(35-5/3.0,6),round(25+5/3.0,6)) in c)
        self.assertTrue((round(5+5/3.0,6),round(15-5/3.0,6)) in c and (round(15+5/3.0,6),round(25-5/3.0,6)) in c and (round(25+5/3.0,6),round(35-5/3.0,6)) in c)

    def test_LMSlSdRcD(self):
        "Local file, Multi core, Single layer, Single date, Single Rectangle, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(5,5))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.62434536,6)


        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.033812345,6)

    def test_LMSlSdIcD(self):
        "Local file, Multi core, Single layer, Single date, Irregular Polygon, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((2.5,2.5),(17.5,2.5),(17.5,17.5),(2.5,17.5))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(10,10))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.350762025,6)


        poly = Polygon(((2.5,2.5),(2.5,37.5),(37.5,37.5),(37.5,2.5)))
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.033812345,6)


        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.31531396,6)

    def test_LMSlSdRCD(self):
        "Local file, Multi core, Single layer, Single date, Single Rectangle, Clip=True, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((0,0),(10,0),(10,10),(0,10))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(5,5))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.62434536,6)

        #pick everything
        poly = Polygon(((0,0),(40,0),(40,40),(0,40)))
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.033812345,6)

    def test_LMSlSdICD(self):
        "Local file, Multi core, Single layer, Single date, Irregular Polygon, Clip=False, Dissolve=True"
        #pick one point in the upper left
        layer = self.singleLayer
        poly = Polygon(((2.5,2.5),(17.5,2.5),(17.5,17.5),(2.5,17.5))) 
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(10,10))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.350762025,6)


        poly = Polygon(((5,5),(5,35),(35,35),(35,5)))
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.124389726,6)


        poly = Polygon(((0,0),(0,10),(30,40),(40,40),(40,30),(10,0)))
        elements = self._access(layer,poly,time,True,True,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(20,20))
        self.assertAlmostEqual(elements[0]['properties']['Prcp'],1.402930205,6)

    def test_LMSlSdMpCD(self):
        "Local file, Single core, Single layer, Single date, Multiple Polygons, Clip=False, Dissolve=True"

        layer = self.singleLayer
        poly = [Polygon(((0,0),(10,0),(10,10),(0,10))), Polygon(((20,20),(30,20),(30,30),(20,30)))]
        time = [datetime.datetime(2000,1,1),datetime.datetime(2000,1,1)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),2)
        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((5,5) in c and (25,25) in c)

        poly = [Polygon(((0,0),(20,0),(20,20),(0,20))), Polygon(((0,10),(20,10),(20,30),(0,30)))]
        elements = self._access(layer,poly,time,True,False,levels,subdivide,subres)

        self.assertEqual(len(elements),2)
        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((10,10) in c and (10,20) in c)

class TestOpenDapNC(unittest.TestCase):

    def _getCtrd(self,element):
        g = element['geometry']
        return (g.centroid.x,g.centroid.y)

    def _access(self,polygons,temporal,dissolve,clip,levels,subdivide,subres):
        POLYINT = polygons
        
        dataset = 'http://hydra.fsl.noaa.gov/thredds/dodsC/oc_gis_downscaling.bccr_bcm2.sresa1b.Prcp.Prcp.1.aggregation.1'

        TEMPORAL = temporal
        DISSOLVE = dissolve
        CLIP = clip
        VAR = 'Prcp'
        kwds = {
            'rowbnds_name': 'bounds_latitude', 
            'colbnds_name': 'bounds_longitude',
            'time_units': 'days since 1950-1-1 00:00:0.0',
        }
        ## open the dataset for reading
        ## make iterable if only a single polygon requested
        if type(POLYINT) not in (list,tuple): POLYINT = [POLYINT]
        ## convenience function for multiple polygons
        elements = ncconv.multipolygon_multicore_operation(dataset,
                                        VAR,
                                        POLYINT,
                                        time_range=TEMPORAL,
                                        clip=CLIP,
                                        dissolve=DISSOLVE,
                                        levels = levels,
                                        ocgOpts=kwds,
                                        subdivide=subdivide,
                                        subres = subres
                                        )

        return elements

    def test_RSSlSdRcd(self):
        "Remote file, Single core, Single layer, Single date, Single Rectangle, Clip=False, Dissolve=False"
        #pick one point in the upper left
        poly = Polygon(((-120.125, 35.875), (-120.0, 35.875), (-120.0, 36.0), (-120.125, 36.0), (-120.125, 35.875)))
        #      POLYGON ((-124.75  25.125,   -67.0   25.125, -67.0  52.875,  -124.75  52.875,   -124.75  25.125))
        time = [datetime.datetime(1960,3,16),datetime.datetime(1960,4,16)]
        levels = None
        subdivide = False
        subres = 'detect'
        elements = self._access(poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),1)
        self.assertEqual(self._getCtrd(elements[0]),(-120.0625,35.9375))

        poly=Polygon(((-120.125, 35.875), (-119.875, 35.875), (-119.875, 36.125), (-120.125, 36.125), (-120.125, 35.875)))
        elements = self._access(poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),4)
        c = [self._getCtrd(x) for x in elements]
        self.assertTrue((-120.0625,35.9375) in c and (-119.9375,35.9375) in c and (-120.0625,36.0625) in c and (-119.9375,36.0625) in c)

    def test_RMSlSdRcd(self):
        "Remote file, Multiple core, Single layer, Single date, Single Rectangle, Clip=False, Dissolve=False"
        #pick one point in the upper left
        poly = None
        time = [datetime.datetime(1960,3,16),datetime.datetime(1960,4,16)]
        levels = None
        subdivide = True
        subres = 'detect'
        elements = self._access(poly,time,False,False,levels,subdivide,subres)

        self.assertEqual(len(elements),76019)

#class testExport(unittest.TestCase):


    #462 222

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()