import ogr

cnt = ogr.GetDriverCount()
for i in range(cnt):
    print(ogr.GetDriver(i).GetName())
