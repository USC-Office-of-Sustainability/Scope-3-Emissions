#Post NAICS assignments: Data processing and Analyses
#congrats- now that you have your NAICS codes for each procurement line item, you can start to analyze your data... or can you?
#Of course you will need to clean up your data more, and if you used a non-dupplicated dataset for predicting the NAICS codes, 
#you need to bind the predicted dataset back to your raw file. 

#FIRST load in your R packages (install first if you haven't already)
library(plyr)
library(dplyr) #careful often plyr will mask dplyr functions, so always uninstall plyr before runing dplyr, or call dplyr for the specific function
library(plotrix) #if you want to get standard errors can use std.error
library(tidyverse) #for cleaning data
library(lubridate) #if you have dates and need to convert to specific date formats
library(stringi) #if you need this for cleaning up text
library(stringr) #if you need this for cleaning up text
#library(stringdist) #probably don't need

#SECOND set your working directory
#something like> setwd("~/Desktop/Sustainability Data and Analyses/Procurement Data") # but you need to edit this based on your filepath

#THIRD- read in our 1:1 emissions files with 1:1 2017-2022 NAICS code matches w/ Co2e emissions

emissions=read.csv("final_NAICS_crosswalk_altered_with_one_to_one_match_with_CO2e_emissions.csv", header=TRUE) 
names(emissions) #need to be able to join this to all the procurement data.. which means that we need to rename the NAICS CODE and DESCRIPTION column names from this file to match the column names from your LLM predict output file. we will do that in a min...

#FOURTH: read in your LLM output file (predicted NAICS datafile)
LLMoutput=read.csv("your_llm_output_with_naics_codes.csv", header=TRUE)
names(LLMoutput)
#will look like:  [1] "Supplier"                   "Line.Item.Description"      "Alt.Spend.Category.Level.2"
#[4] "ML_pred1_NAICS"             "ML_pred1_confidence"        "ML_pred1_description"      
#[7] "ML_pred2_NAICS"             "ML_pred2_confidence"        "ML_pred2_description"      
#[10] "ML_pred3_NAICS"             "ML_pred3_confidence"        "ML_pred3_description"      
#[13] "ML_model_used"  

#note we assume you already manually cleaned this file- meaning you looked at all the low confidence rows for ML_pred1_confidence and manually corrected them.. technically we would suggest creating a new column w/ the corrections.. .but this code assumes you used the same column name (ML_pred1_NAICS), so change that if you have a final_naics_code column regarding the below

#FIFTH- rename the column names in the LLM output file for an easy join process w/ the emissions file
LLMoutclean <- LLMoutput %>%
  rename(chosen_2022_code = ML_pred1_NAICS, chosen_2022_title=ML_pred1_description)
names(LLMoutclean)

#SIXTH- BIND DATA #USING THE PLYR LIBRARY.. 

LLMoutput_with_emissions=left_join(LLMoutclean, emissions)
dim(procurement_with_emissions)
dim(LLMoutclean) #double check that your joined file matches your original llm output file.. 

#SEVENTH! BIND RESULTING DATA BACK TO YOUR RAW FILE (IMPORTANT IF YOU DEDUPLICATED BY LINE ITEM PREVIOUSLY)
#if you had an non-duplicated datafile that you sent to the NAICS app, then you need to rejoin this to your raw file by Line.Item.Description. so for example, and make sure that your column names and contents are in the same case (all uppercase, all title case or all lowercase... )

raw_procurementdata=read.csv("your_raw_procurementdata.csv", header=TRUE)
#lets say some of your data columns are as follows #LINE.ITEM.DESCRIPTION	#ALT.SPEND.CATEGORY.LEVEL.2	#SUPPLIER	

#below are some cleaning steps to do on both dataframes before joining if you need to join back to raw file..

# helper normalizer (base R, no extra packages required)
normalize_text_base <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- NA_character_
  # remove punctuation, collapse whitespace, trim, lowercase
  x <- gsub("[[:punct:]]+", " ", x)
  x <- gsub("\\s+", " ", x)
  x <- trimws(x)
  x <- tolower(x)
  x
}

# helper functions (stringi-based)
safe_to_utf8 <- function(x) {
  x <- as.character(x)
  na_idx <- is.na(x)
  y <- stringi::stri_enc_toutf8(x)   # repair/ensure UTF-8
  y[na_idx] <- NA_character_
  y
}

normalize_text_stringi <- function(x) {
  x <- safe_to_utf8(x)
  
  x <- stringi::stri_trans_nfkc(x)
  x <- stringi::stri_trans_general(x, "Any-Latin; Latin-ASCII")
  
  # remove trademark-like symbols and replacement chars
  x <- stringi::stri_replace_all_regex(x, "[™®©\uFFFD]", " ")
  
  # remove control chars
  x <- gsub("[[:cntrl:]]", " ", x)
  
  # keep only letters/numbers/spaces for joining
  x <- gsub("[^A-Za-z0-9]+", " ", x)
  x <- stringr::str_squish(x)
  
  x
}

# Full safe pipeline using stringi (no base grepl)
cleaned_raw <- raw_procurementdata %>%
  rename_with(toupper) %>%
  mutate(across(where(is.factor), as.character)) %>%
  mutate(across(where(is.character),
                ~ {
                  col <- safe_to_utf8(.x)
                  is_na <- is.na(col)
                  matched_letters <- stringi::stri_detect_regex(col, "[A-Za-z]")
                  out <- ifelse(is_na, NA_character_,
                                ifelse(matched_letters, stringi::stri_trans_toupper(col), col))
                  out
                })) %>%
  mutate(
    LINE.ITEM.DESCRIPTION_N = normalize_text_stringi(LINE.ITEM.DESCRIPTION)
  ) #you will bind the two dataframes by the LINE.ITEM.DESCRIPTION_N


cleaned_LLMoutput <- LLMoutput_with_emissions %>%
  rename_with(toupper) %>%
  mutate(across(where(is.factor), as.character)) %>%
  mutate(across(where(is.character),
                ~ {
                  col <- safe_to_utf8(.x)
                  is_na <- is.na(col)
                  matched_letters <- stringi::stri_detect_regex(col, "[A-Za-z]")
                  out <- ifelse(is_na, NA_character_,
                                ifelse(matched_letters, stringi::stri_trans_toupper(col), col))
                  out
                })) %>%
  mutate(
    LINE.ITEM.DESCRIPTION_N = normalize_text_stringi(Line.Item.Description)) #see how this was Titlecase and not all caps from this dataframe? Attention to detail in R is super important  

#you will bind the two dataframes by the LINE.ITEM.DESCRIPTION_N

All_procurement_data_emissions=left_join(cleaned_raw, cleaned_LLMoutput, by="LINE.ITEM.DESCRIPTION_N")

#double check that lenght of dataframe is the same as the raw dataframe length

dim(cleaned_raw)
dim(All_procurement_data_emissions) #they should match

write.csv(All_procurement_data_emissions, "final_procurement_alldata_with_NAICSemissions.csv", row.names=TRUE) #congrats, now make sure you look at the output and check that there aren't NAS.. some things might have gone wrong, so see some tips below for what might have happened

#DOUBLE CHECK THESE DATA BEFORE ANALYZING/SUMMARIZING!

#no need to proceed unless you have errors with previous NAICS codes being 2017 codes and descriptions instead of 2022 codes and descriptions (can happen from manual assignments)
#below are tips for if this is your issue also.. you will have to customize this based on your data- but this is an idea of what to do.. 

all= read.csv("final_procurement_alldata_with_NAICSemissions.csv", header=TRUE) #not this is not cleaned- some NAICS codes are bad due to errors the LLM learned from wrongly assigned NAICS codes (manual assignments had some mistakes, and sometimes used 2017 codes..)
names(all)
#for example, in our data file we needed to make some replacements that were errors from manual categorization that the LLM repeated mistakes with
#PRIMARY BATTERY MANUFACTURING  #REPLACE WITH 335910	"BATTERY MANUFACTURING"
#PAINT AND COATING MANUFACTURING #replace WITH 325510 "PAINT AND COATING MANUFACTURING"
#METAL HOUSEHOLD FURNITURE MANUFACTURING #replace with 337126	"HOUSEHOLD FURNITURE (EXCEPT WOOD AND UPHOLSTERED) MANUFACTURING"
#INDUSTRIAL MACHINERY MANUFACTURING #replace with 811310	"COMMERCIAL AND INDUSTRIAL MACHINERY AND EQUIPMENT (EXCEPT AUTOMOTIVE AND ELECTRONIC) REPAIR AND MAINTENANCE"

all$NAICS.2022.CODE.FINAL
all2 <- all %>%
  mutate(
    desc_key = str_to_upper(str_squish(as.character(chosen_2022_code))),
   chosen_2022_code = case_when(
      desc_key == "PRIMARY BATTERY MANUFACTURING" ~ "335910",
      desc_key == "PAINT AND COATING MANUFACTURING" ~ "325510",
      desc_key == "METAL HOUSEHOLD FURNITURE MANUFACTURING" ~ "337126",
      desc_key == "INDUSTRIAL MACHINERY MANUFACTURING" ~ "811310",
      TRUE ~ as.character(chosen_2022_code)
    ),
    chosen_2022_title = case_when(
      desc_key == "PRIMARY BATTERY MANUFACTURING" ~ "BATTERY MANUFACTURING",
      desc_key == "PAINT AND COATING MANUFACTURING" ~ "PAINT AND COATING MANUFACTURING",
      desc_key == "METAL HOUSEHOLD FURNITURE MANUFACTURING" ~ "HOUSEHOLD FURNITURE (EXCEPT WOOD AND UPHOLSTERED) MANUFACTURING",
      desc_key == "INDUSTRIAL MACHINERY MANUFACTURING" ~ "COMMERCIAL AND INDUSTRIAL MACHINERY AND EQUIPMENT (EXCEPT AUTOMOTIVE AND ELECTRONIC) REPAIR AND MAINTENANCE",
      TRUE ~ as.character(chosen_2022_title)
    ), chosen_2022_code=as.integer(chosen_2022_code)
  ) %>%
  select(-desc_key)

# lookup table keyed by 2017 code, but carrying the 2022 code/desc from emissions file
names(emissions)
em_2017 <- emissions %>%
  transmute(
    chosen_2017_code,
    em_without_2017 = em_without,
    em_with_2017    = em_with
  )

head(em_2017)
cleaned_emissions_final <- all2 %>%
  left_join(emissions, by = "chosen_2022_code") %>%
  left_join(em_2017, by = c("chosen_2022_code" = "chosen_2017_code")) %>%
  mutate(
    match_flag = case_when(
      !is.na(em_without) | !is.na(em_with) ~ "matched on 2022 code",
      !is.na(em_without_2017) | !is.na(em_with_2017) ~ "matched via 2017 code",
      TRUE ~ "unmatched"
    ),
    em_without = coalesce(em_without, em_without_2017),
    em_with    = coalesce(em_with, em_with_2017)
  ) %>%
  select(
    -em_without_2017,
    -em_with_2017
  )
write.csv(cleaned_emissions_final, "CleanedALL_Procurement_with_emissions.csv", row.names=FALSE)
#double check this file.. 
