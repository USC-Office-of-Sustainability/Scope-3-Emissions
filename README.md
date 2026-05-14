# Scope-3-Emissions (Category 1)
At the University of Southern California (USC), our Office of Sustainability is working to automate many of the tedious data processes required for sustainabilty reporting. This includes reporting out our Scope 3 GHG emissions. For each of our automation processes, we also conduct many manual checks and suggest that you do the same as well to improve the quality of your final datasets. Here, we will be sharing our code and tools (apps) that help to automate emission factors for the following Scope 3 Categories:

***Category 1: Purchased Goods and Services

[Python code and app for automating NAICS code categorization of non-food items (Office supplies, Lab, IT/Hardware, Furniture, etc)](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/tree/main/NAICS_Categorization_App)

[Rscript and datasets for creating a 1:1 NAICS 2017-2022 crosswalk with 2022 emission factors](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/tree/main/NAICS%20Crosswalk%20%26%20Emissions)

[Rscript data post-processing of LLM NAICS output file, binding to emissions data, and then binding these back to your procurement file (if you deduplicated by lineitem)](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/blob/main/Data%20Processing%20and%20Analysis/DataBinding_before_Analyses_forGithub.R#:~:text=DataBinding_before_Analyses_forGithub.R)

Code for automating food weights: (coming soon).  We suggest using Rebecca Grekin's [TasteFood tool](https://tastefood.org) for Food Categorization

***Category 7: Commuting Emissions 
Code and files coming soon
