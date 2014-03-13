rm(list = ls())
require(ggplot2)
require(reshape)
require(plyr)

IN_DATA = '/home/local/WX/ben.koziol/Downloads/qed_city_centroids_duration/qed_city_centroids_duration.csv'

## read in data from csv files
data = read.csv(IN_DATA)

## we only want the duration data
data = subset(data,CALC_NAME == 'Duration')

## only look at july
data = subset(data,MONTH == 7)

## map ugids
data$CITY_NAME[data$UGID == 1] = 'Boston, MA'
data$CITY_NAME[data$UGID == 2] = 'Indianapolis, IN'
data$CITY_NAME[data$UGID == 3] = 'Washington, DC'

## this is the y-axis
data$TIME = as.Date(paste(data$YEAR,data$MONTH,15,sep='-'))

p = ggplot(data=data,aes(x=TIME,y=VALUE,color=ALIAS,fill=ALIAS))
p = p + geom_bar(stat='identity',position='dodge')
p = p + labs(title='Average Duration of Heat Spells >= 32C in July',x='Time',y='Average Duration (days)')
p = p + facet_grid(CITY_NAME~.)
print(p)
