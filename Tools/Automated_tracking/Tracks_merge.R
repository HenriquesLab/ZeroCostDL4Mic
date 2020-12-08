
library(readr)
library(ggplot2)
library(RColorBrewer) 
library(reshape2) 
library(gridExtra)
library(tools)
library(plyr)
library(matrixStats)
library(tidyverse)
library(stringr)


path <- file.choose()
DIR  <- dirname(path)

#Merge the track basic info

Track_results <- list.files(path = DIR,     # Identify all csv files in folder
                            pattern = "tracks_properties.csv", full.names = TRUE) %>% 
  lapply(read_csv, col_types = cols(.default = "c")) %>%                                            # Store all files in list
  bind_rows                                                       # Combine data sets into one data set 

write.csv(Track_results, file=paste(DIR, "track_Results.csv", sep=""), row.names = T)

#Merge the full tracks

dir.create(file.path(DIR, "single_in"))
DIROUT_single_IN <- file.path(DIR, "single_in", "") 
dir.create(file.path(DIR, "merge"))
DIROUT_merge <- file.path(DIR, "merge", "") 

fileNames <- dir(DIR, full.names=TRUE, pattern ="spots_properties.csv")

for (fileName in fileNames) {
  p = file_path_sans_ext(basename(fileName))
  Values <- read_csv(fileName)
  MaxValue <- max(Values$"Tracking ID", na.rm = TRUE)
  #dp <- filter(Values, Values$"Tracking ID" == 1)
  #write.csv(dp, file=paste(DIROUT_single_IN,p,"_", 1, ".csv", sep=""), row.names = F)


  i = 0
  repeat {
    dp <- filter(Values, Values$"Tracking ID" == i)
    j = i +1
    write.csv(dp, file=paste(DIROUT_single_IN,p,"_", i, ".csv", sep=""), row.names = F)
    i = i +1
    if (i == MaxValue +1) {
      break
    }
  }
  
}

# Read all the track and add a column with track numbers

fileNames <- dir(DIROUT_single_IN, full.names=TRUE, pattern =".csv")


p = 0
for (fileName in fileNames) {
  name = file_path_sans_ext(basename(fileName))
  individual_tracks <- read_csv(fileName)
  NumberofRow <- nrow(individual_tracks)
  vec <- rep(c(p), times = NumberofRow)
  individual_tracks$Track_number <- vec
  write.csv(individual_tracks, file=paste(DIROUT_merge,name,"_", ".csv", sep=""), row.names = F)
  p = p+1
  
}


# Merge all the files
Track_merge <- list.files(path = DIROUT_merge,     # Identify all csv files in folder
                            pattern = ".csv", full.names = TRUE) %>% 
  lapply(read_csv, col_types = cols(.default = "c")) %>%                                            # Store all files in list
  bind_rows                                                       # Combine data sets into one data set 

write.csv(Track_merge, file=paste(DIR, "track_Merge.csv", sep=""), row.names = T)





