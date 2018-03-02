rm(list = ls())
require(ggplot2)
require(reshape)
require(plyr)

IN_DATA = '/home/local/WX/ben.koziol/Dropbox/nesii/docs/NOAA-WriteUp-20140317/ocgis_output/ocgis_output.csv'
IN_GEOM_DATA = '/home/local/WX/ben.koziol/Dropbox/nesii/docs/NOAA-WriteUp-20140317/ocgis_output/shp/ocgis_output_ugid.csv'

## read in data from csv files
data = read.csv(IN_DATA)
data.geom = read.csv(IN_GEOM_DATA,sep=';')

## convert time to objects from string
data$TIME = as.Date(data$TIME)
## join the output data file with the geometry descriptions
data.joined = merge(data,data.geom,by="UGID",all=TRUE)

## data.joined = subset(data.joined,WATERSHED %in% c('ALAFIA RIVER','COASTAL CREEKS','BULLFROG CREEK'))

## this function calculates some standard statistics on the output data for each
## product grouped by watershed and time (statistic applied across grid cells)
# wstats = function(df){
#   attach(df)
#   ret = list()
#   ret['sd'] = sd(VALUE)
#   ret['mean'] = mean(VALUE)
#   ret['median'] = median(VALUE)
#   ret['min'] = min(VALUE)
#   ret['max'] = max(VALUE)
#   ret['n'] = length(VALUE)
#   ret['sum'] = sum(VALUE)
#   detach(df)
#   return(as.data.frame(ret))
# }

# data.stats = ddply(data.joined,.(WATERSHED,TIME,VARIABLE,ALIAS),wstats)

p = ggplot(data=data.joined,aes(x=TIME,y=VALUE))
p = p + geom_line(aes(color=VARIABLE),size=0.25)
p = p + labs(title='Spatially Averaged Temperature Simulations \nby NOAA Climate Division in Colorado',
             x='Year (2010-2021)',
             y='Simulated Temperature (Celsius)')
p = p + facet_grid(NAME~.)
print(p)