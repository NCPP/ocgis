rm(list = ls())
require(plyr)
require(foreign)


CSV_DATA = '/home/local/WX/ben.koziol/Dropbox/nesii/conference/FOSS4G_2013/figures/heat_index/urban_areas_2000_heat_index/urban_areas_2000_heat_index.csv'
SHP_PATH = '/home/local/WX/ben.koziol/Dropbox/nesii/conference/FOSS4G_2013/figures/heat_index/urban_areas_2000/urban_areas_2000.dbf'
SHP_2011_PATH = '/home/local/WX/ben.koziol/Dropbox/nesii/conference/FOSS4G_2013/figures/heat_index/urban_areas_2000/urban_areas_2000_2011.dbf'
SHP_2020_PATH = '/home/local/WX/ben.koziol/Dropbox/nesii/conference/FOSS4G_2013/figures/heat_index/urban_areas_2000/urban_areas_2000_2020.dbf'
HI_THRESHOLD = 130
MONTH = 7


data = read.csv(CSV_DATA)
colnames(data) = toupper(colnames(data))
data.2011 = subset(data,data$YEAR == 2011 & data$VALUE > HI_THRESHOLD)
data.2020 = subset(data,data$YEAR == 2020 & data$VALUE > HI_THRESHOLD)

## read the shapefile dbf backing it up first
dbf = read.dbf(SHP_PATH)

## join the calculated data to it
dbf.2011 = join(dbf,data.2011,by="UGID")
dbf.2020 = join(dbf,data.2020,by="UGID")

## write it out
write.dbf(dbf.2011,SHP_2011_PATH)
write.dbf(dbf.2020,SHP_2020_PATH)
