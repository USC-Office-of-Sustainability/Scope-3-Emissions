#tips for preprocessing your data for training an LLM model or for predicting outputs
#to train a new LLM model and then use it to predict NAICS codes on unseen data, we recommend removing duplicate line-item descriptions from your original raw datafiles so that the processing time and cost is less (also less GHG emissions!)

#for a training dataset, we recommend finding the most common assigned NAICS code for each line item description before deduplicating (if you had them manually assigned) as likely the most common NAICS code will be the more accurate naics code

#install and load in the below packages; 
library(tidyverse) #includes mutate function and other data cleaning functions
library(dplyr) # note dplyr and plyr conflict with one another, so if you are running plyr on your laptop w/ other datasets/code then you will want to detach plyr before running dplyr functions, or specifically call dplyr with your function. 

#set your working directory (file path) see tips and how-to below
#https://www.sthda.com/english/wiki/running-rstudio-and-setting-up-your-working-directory-easy-r-programming

#read in your dataset: 
proc_manual_naics=read.csv("your_procurment_dataset_with_manual_NAICS_codes.csv", header=TRUE)

#run and use the below function first

most_common <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) return(NA)
  names(which.max(table(x)))
}

#now lets deduplicate things
#check the column names of your file- they might be different than my example
names(proc_manual_naics)

#my example column names as follows: 
#"LINE.ITEM.DESCRIPTION"     # "SUPPLIER"      
#"NAICS.2022.CODE.FINAL"      #"NAICS.2022.DESC.FINAL"     
#"chosen_2022_title"         # "chosen_2017_code"          
# "chosen_2017_title"       
levels(as.factor(proc_manual_naics$LINE.ITEM.DESCRIPTION)) #TO GET A SNEAK PEEK

detach(package:plyr) #just in case you have it loaded. if you get an error it just means it wasn't loaded which is good right now as it conflicts w/ dplyr functions
proc_no_dup<-proc_manual_naics%>%
  group_by(LINE.ITEM.DESCRIPTION) %>%
  mutate(
    common_desc = most_common(NAICS.2022.DESC.FINAL),
    common_code = most_common(NAICS.2022.CODE.FINAL)
  ) %>%
  filter(
    NAICS.2022.DESC.FINAL== common_desc,
    NAICS.2022.CODE.FINAL == common_code
  ) %>%
  slice(1) %>%   # in case multiple rows match
  ungroup() %>%
  select(-common_desc, -common_code)

dim(proc_no_dup) #should be a much smaller dataframe than your original dataframe. 

#write this out as a csv file: 
write.csv(proc_no_dup, 'Training_dataset_for_LLM_no_dups_common_naics.csv', row.names=FALSE)

#now if you want to predict data on non NAICS classified data, we suggest you also deduplicate your data to make runtime faster... but you don't need to find the most common code. 

need_NAICS_data=read.csv("procurement_data_need_naics.csv", header=TRUE)
#make sure this file is in the same working directory that you are calling files from.. 

#lets say your columns are named as follows:
#"LINE.ITEM.DESCRIPTION"     # "SUPPLIER"     #"FISCAL_YEAR" #"ALT.SPEND.CATEGORY.LEVEL.2"

#LETS REMOVE DUPLICATES based on line item description
need_naics_nodup<- need_NAICS_data%>%
  distinct(LINE.ITEM.DESCRIPTION, .keep_all = TRUE)  

#or if you have the same line item descriptions but different suppliers then maybe you want to include SUPPLIER in the above code as well which would like like: 
#need_naics_nodup<- need_NAICS_data%>%distinct(LINE.ITEM.DESCRIPTION,SUPPLIER, .keep_all = TRUE)  
#export your file so you can use it in our NAICS classifier app: 
write.csv(need_naics_nodup, "Procurement_data_no_duplicates_need_NAICS_codes.csv", row.names=FALSE)

#important, you will eventually want to bind the resulting predicted dataset back to your RAW procurement file so that you match NAICS codes to every row, and you dont' just analyze the deduplicated dataset... 
