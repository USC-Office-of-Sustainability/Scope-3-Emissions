# Binding your procurement data with NAICS emissions
After you obtain your NAICS-categorized procurement dataset, you will need to bind these data to an emissions file that is matched with 2022 NAICS codes. 

This entails several pre-processing datasteps: 
1. **Create a 1:1 match between 2017 and 2022 NAICS codes**
2.   - Currently the 2022 Co2e emissions are matched to 2017 NAICS codes. Thus you need to create a 1:1 match to 2022 NAICS codes from the 2017 NAICS codes as we used 2022 NAICS codes in categorizing procurement data.
     - The original 2017-2022 crosswalk file has a many-many match issue, so we resolve this. 
3. **Bind the 1:1 match NAICS Co2e emissions datafile to your procurement data**

---

## 1. Create a 1:1 match between the 2017 and 2022 NAICS codes: 
[Use this Rscript](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/new/main/NAICS%20Crosswalk%20%26%20Emissions#:~:text=NAICS_crosswalk_emissions_code_to_share.R)
[Use this csv file](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/new/main/NAICS%20Crosswalk%20%26%20Emissions#:~:text=2022%2D2017%2DNAICS%2D-,Code,-%2DConcordance%2D1.csv)

[Or skip this step and just use the final output file here](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/new/main/NAICS%20Crosswalk%20%26%20Emissions#:~:text=final_NAICS_crosswalk_altered_with_one_to_one_match_with_CO2e_emissions.csv)

Sources of original files: 
[Original CO2e and GHG emissions files](https://catalog.data.gov/dataset/supply-chain-greenhouse-gas-emission-factors-v1-3-by-naics-6)
[Original crosswalk file with many: many matches between 2017-2022 NAICS codes](https://www.naics.com/2022-naics-changes/)

## 2. Bind the 1:1 emissions file to your procurement file using the 2022 NAICS codes

[Rscript to help with post-data processing](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/blob/main/Data%20Processing%20and%20Analysis/DataBinding_before_Analyses_forGithub.R#:~:text=DataBinding_before_Analyses_forGithub.R)

Questions? Email: oosdata(at)usc.edu
