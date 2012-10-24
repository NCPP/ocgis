rm(list = ls())
require(plyr)
require(foreign)

## function for binning
bin = function(dfr,bounds){
  ndays = nrow(dfr)
  percents = list()
  dfr = subset(dfr,!is.na(VALUE))
  for (name in names(bounds)){
    vec = bounds[[name]]
    percents[name] = round((sum(dfr$VALUE >= vec[1] & dfr$VALUE <= vec[2])/ndays)*100,2)
  }
  return(as.data.frame(percents))
}

## some classification variables
bounds = list(caution=c(80,90),
              ex_caution=c(91,105),
              danger=c(106,130),
              ex_danger=c(131,99999))

## load time, calculation, user geometry, and value data.frames
tgid = read.csv('tgid.csv')
cid = read.csv('cid.csv')
value = read.csv('value.csv')

## remove empties and select heat index
value = subset(value,CID == 2)

## join data to time variables for temporal grouping
value = join(tgid,value,by="TGID")

## calculate percentages
by_year = ddply(value,c("YEAR","UGID"),bin,bounds)
by_state = ddply(value,c("UGID"),bin,bounds)

## read the shapefile dbf backing it up first
file.copy('state_boundaries/state_boundaries.dbf','state_boundaries/state_boundaries.dbf.backup')
state_dbf = read.dbf('state_boundaries/state_boundaries.dbf')

## join the calculated data to it
state_dbf = join(state_dbf,by_state,by="UGID")

## write it out
write.dbf(state_dbf,'state_boundaries/state_boundaries.dbf')