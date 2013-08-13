rm(list = ls())
require(ggplot2)
require(plyr)

IN_DATA = '/home/local/WX/ben.koziol/Downloads/tbw_example/tbw_example.csv'
IN_GEOM_DATA = '/home/local/WX/ben.koziol/Downloads/tbw_example/shp/tbw_example_ugid.csv'

## read in data from csv files
data = read.csv(IN_DATA)
data.geom = read.csv(IN_GEOM_DATA)

## convert time to objects from string
data$TIME = as.Date(data$TIME)
## masked data values are returned, we do not want those data values for this example
data.nomask = subset(data,data$VALUE != 1e20)
## join the output data file with the geometry descriptions
data.joined = merge(data.nomask,data.geom,by="UGID",all=TRUE)
## remove extra GIDs value data to account for cells overlapping watersheds
data.unique = data.nomask
data.unique$UGID = NULL
data.unique = unique(data.unique)
p90 = quantile(data.unique$VALUE,0.9)

data.joined = subset(data.joined,WATERSHED %in% c('ALAFIA RIVER','COASTAL CREEKS','BULLFROG CREEK'))

## this function calculates some standard statistics on the output data for each
## product grouped by watershed and time (statistic applied across grid cells)
wstats = function(df){
  attach(df)
  ret = list()
  ret['sd'] = sd(VALUE)
  ret['mean'] = mean(VALUE)
  ret['median'] = median(VALUE)
  ret['min'] = min(VALUE)
  ret['max'] = max(VALUE)
  ret['n'] = length(VALUE)
  ret['sum'] = sum(VALUE)
  ret['p90'] = quantile(VALUE,0.9)
  detach(df)
  return(as.data.frame(ret))
}

data.stats = ddply(data.joined,.(WATERSHED,TIME,VARIABLE,ALIAS),wstats)
data.stats$Legend = data.stats$ALIAS

p = ggplot(data=data.stats,aes(x=TIME,y=sd))
p = p + geom_line(aes(color=Legend,linetype=Legend),size=0.75)
p = p + labs(title='Standard Deviation by Watershed (mm/day)',x='Time',y='Precipitation (mm/day)')
p = p + facet_grid(WATERSHED~.)
p = p + theme(legend.text=element_text(size=16),axis.text.x=element_text(size=16),axis.text.y=element_text(size=16))
p = p + theme(axis.title.x=element_text(size=18),axis.title.y=element_text(size=18))
p = p + theme(plot.title=element_text(size=19),strip.text.y=element_text(size=16))
p = p + theme(legend.title=element_text(size=16))
print(p)
