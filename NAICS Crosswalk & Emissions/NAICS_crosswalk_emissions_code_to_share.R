#Julie Hopper's R code - fixing the 2022-2017 NAICS cross walk file, so we can bind to the emissions file that uses 2017 NAICS codes... and then that way I can bind back to 2022 NaIcs codes in our dataframes..
#set working directory (will be different on other people's computers)
setwd("~/Dropbox/Sustainability Data and Analyses/Procurement Data (food, electronics, lab)")

#load in R packages needed for this to run
library(dplyr)
library(tidyr)
library(stringdist)
library(fuzzyjoin)
#install.packages("janitor")
library(janitor)

# --- read in files ---------------------------------------------
crosswalk<-read.csv("2022-2017-NAICS-Code-Concordance-1.csv", header=TRUE)
emissions <- read.csv("SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv", header=TRUE) 
#explore the data first...
names(crosswalk)
names(emissions)
emissions$Supply.Chain.Emission.Factors.with.Margins
emissions$Supply.Chain.Emission.Factors.without.Margins
class(emissions$Supply.Chain.Emission.Factors.with.Margins)
class(emissions$Supply.Chain.Emission.Factors.without.Margins)
class(emissions$X2017.NAICS.Title)
# Ensure emissions 2017 code is character
emissions <- emissions %>%
  mutate(  `X2017.NAICS.Code` = as.character(`X2017.NAICS.Code`))
names(crosswalk)
crosswalk$X2017.NAICS.Code.1
# Ensure crosswalk codes/titles are character
crosswalk <- crosswalk %>%
  mutate(
    `X2022.NAICS.Code` = as.character(`X2022.NAICS.Code`),
    `X2022.NAICS.Title` = as.character(`X2022.NAICS.Title`),
    `X2017.NAICS.Code`  = as.character(`X2017.NAICS.Code`),
    `X2017.NAICS.Title`= as.character(`X2017.NAICS.Title`)
  )

#below- alternative method of finding a best match based on fuzzy joining within already predetermined matches etc for where there is many to many or one to many or many to one relations. 

# Produce final one-to-one mapping (2022 -> chosen 2017) and attach emissions
# Use exact column names. No renaming of original columns.
#
# Inputs required (in memory):
#  - crosswalk: columns include X2022.NAICS.Code, X2022.NAICS.Title, X2017.NAICS.Code, X2017.NAICS.Title
#  - emissions: columns include X2017.NAICS.Code, X2017.NAICS.Title (optional), Supply.Chain.Emission.Factors.without.Margins, Supply.Chain.Emission.Factors.with.Margins
#
# Output:
#  - final_one_to_one_match_with_emissions.csv (one row per chosen 2022 code, with chosen 2017 and emissions)

# ---- helpers ----
safe_to_utf8 <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- NA_character_
  out <- iconv(x, from = "UTF-8", to = "UTF-8", sub = "")
  bad <- is.na(out)
  if (any(bad)) out[bad] <- iconv(x[bad], from = "latin1", to = "UTF-8", sub = "")
  bad2 <- is.na(out)
  if (any(bad2)) out[bad2] <- gsub("[^[:print:]]+", "", x[bad2])
  out
}
canon <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x_utf8 <- safe_to_utf8(x)
  if (requireNamespace("stringi", quietly = TRUE)) x_utf8 <- stringi::stri_trans_general(x_utf8, "Latin-ASCII")
  x_norm <- tolower(x_utf8)
  x_norm <- gsub("[^a-z0-9 ]+", " ", x_norm)
  x_norm <- gsub("\\s+", " ", x_norm)
  trimws(x_norm)
}
coerce_numeric_safely <- function(x){
  x <- as.character(x)
  x[is.na(x) | x == ""] <- NA_character_
  x <- gsub(",", "", x)
  x <- gsub("\\(", "-", x)
  x <- gsub("\\)", "", x)
  suppressWarnings(as.numeric(x))
}

# ---- 0. Ensure types and make a defensive per-2017 emission summary if there are multiple rows per 2017 code ----
# (If your emissions table already has one row per 2017 code, this is effectively identity)
emissions <- emissions %>%
  mutate(
    `X2017.NAICS.Code` = as.character(`X2017.NAICS.Code`),
    `X2017.NAICS.Title` = as.character(`X2017.NAICS.Title`),
    `Supply.Chain.Emission.Factors.without.Margins` = coerce_numeric_safely(`Supply.Chain.Emission.Factors.without.Margins`),
    `Supply.Chain.Emission.Factors.with.Margins`    = coerce_numeric_safely(`Supply.Chain.Emission.Factors.with.Margins`)
  )

# Defensive: collapse multiple emissions rows per 2017 code by taking mean (keeps final join 1:1)
# If emissions is guaranteed 1:1, this will simply copy the unique values.
em_for_join <- emissions %>%
  group_by(`X2017.NAICS.Code`) %>%
  summarize(
    em_without = if(all(is.na(`Supply.Chain.Emission.Factors.without.Margins`))) NA_real_ else mean(`Supply.Chain.Emission.Factors.without.Margins`, na.rm = TRUE),
    em_with    = if(all(is.na(`Supply.Chain.Emission.Factors.with.Margins`))) NA_real_ else mean(`Supply.Chain.Emission.Factors.with.Margins`, na.rm = TRUE),
    em_2017_title_from_em = first(na.omit(`X2017.NAICS.Title`)),
    .groups = "drop"
  )

# ---- 1. Build distinct candidate pairs from the crosswalk (use crosswalk pairs exactly) 

# --- safe join + effective title creation (use this block instead of the failing mutate) ---
cw_pairs <- crosswalk %>%
  transmute(`X2022.NAICS.Code`, `X2022.NAICS.Title`, `X2017.NAICS.Code`, `X2017.NAICS.Title`) %>%
  distinct() %>%
  left_join(
    em_for_join %>% select(`X2017.NAICS.Code`, em_2017_title_from_em),
    by = "X2017.NAICS.Code"
  ) %>%
  # create UTF-8 safe temporary copies BEFORE any string ops
  mutate(
    X2017.NAICS.Title_safe = safe_to_utf8(`X2017.NAICS.Title`),
    em_2017_title_from_em_safe = safe_to_utf8(em_2017_title_from_em)
  ) %>%
  # now create effective_2017_title safely using the safe copies
  mutate(
    effective_2017_title = ifelse(
      nzchar(trimws(X2017.NAICS.Title_safe)),
      X2017.NAICS.Title_safe,
      em_2017_title_from_em_safe
    )
  ) %>%
  # optionally drop the helper safe columns (keep original X2017.NAICS.Title intact)
  select(-X2017.NAICS.Title_safe, -em_2017_title_from_em_safe)

# ---- 2. Identify ambiguous groups (both directions) ----
ambig_2022 <- cw_pairs %>% group_by(`X2022.NAICS.Code`) %>% summarize(n_2017 = n_distinct(`X2017.NAICS.Code`), .groups="drop") %>% filter(n_2017 > 1) %>% pull(`X2022.NAICS.Code`)
ambig_2017 <- cw_pairs %>% group_by(`X2017.NAICS.Code`) %>% summarize(n_2022 = n_distinct(`X2022.NAICS.Code`), .groups="drop") %>% filter(n_2022 > 1) %>% pull(`X2017.NAICS.Code`)

cw_pairs <- cw_pairs %>% mutate(ambiguous = (`X2022.NAICS.Code` %in% ambig_2022) | (`X2017.NAICS.Code` %in% ambig_2017))

# ---- 3. Canonicalize titles safely ----
cw_pairs <- cw_pairs %>%
  mutate(
    title_2022_c = canon(`X2022.NAICS.Title`),
    title_2017_eff_c = canon(effective_2017_title)
  )

# ---- 4. Compute similarity (Jaro-Winkler). We'll use similarity only to resolve ambiguous groups ----
method_used <- "jw"
cw_pairs <- cw_pairs %>%
  mutate(
    title_dist = ifelse(title_2017_eff_c == "" , Inf, stringdist(title_2022_c, title_2017_eff_c, method = method_used)),
    title_sim  = ifelse(is.infinite(title_dist), NA_real_, 1 - (title_dist / (pmax(nchar(title_2022_c), nchar(title_2017_eff_c)) + 1)))
  )

# Save candidate diagnostics (optional)
write.csv(cw_pairs %>% arrange(desc(ambiguous), `X2022.NAICS.Code`, title_dist), "candidate_scores_all_pairs.csv", row.names = FALSE)
write.csv(cw_pairs %>% filter(ambiguous) %>% arrange(`X2022.NAICS.Code`, title_dist), "ambiguous_candidate_pairs.csv", row.names = FALSE)

# ---- 5. Compose final 1:1 mapping
# Strategy:
#  - Keep unambiguous pairs exactly as they are (they must remain)
#  - For ambiguous pairs, pick best candidate edges by greedy highest title_sim ensuring 1:1 constraints
keep_unambig <- cw_pairs %>%
  filter(!ambiguous) %>%
  distinct(`X2022.NAICS.Code`, `X2022.NAICS.Title`, `X2017.NAICS.Code`, effective_2017_title) %>%
  transmute(
    chosen_2022_code = `X2022.NAICS.Code`,
    chosen_2022_title = `X2022.NAICS.Title`,
    chosen_2017_code = `X2017.NAICS.Code`,
    chosen_2017_title = effective_2017_title,
    title_sim = NA_real_,
    title_dist = NA_real_,
    source = "unambig"
  )

edge_list <- cw_pairs %>%
  filter(ambiguous) %>%
  mutate(title_sim_sort = ifelse(is.na(title_sim), -Inf, title_sim)) %>%
  arrange(desc(title_sim_sort)) %>%
  select(`X2022.NAICS.Code`, `X2022.NAICS.Title`, `X2017.NAICS.Code`, effective_2017_title, title_sim, title_dist) %>%
  rename(
    chosen_2022_code = `X2022.NAICS.Code`,
    chosen_2022_title = `X2022.NAICS.Title`,
    chosen_2017_code = `X2017.NAICS.Code`,
    chosen_2017_title = effective_2017_title
  )

# greedy selection
assigned_2022 <- unique(keep_unambig$chosen_2022_code)
assigned_2017 <- unique(keep_unambig$chosen_2017_code)
selected_edges <- list()
if (nrow(keep_unambig) > 0) selected_edges[[length(selected_edges) + 1]] <- keep_unambig

for (i in seq_len(nrow(edge_list))) {
  r <- edge_list[i, , drop = FALSE]
  c22 <- as.character(r$chosen_2022_code)
  c17 <- as.character(r$chosen_2017_code)
  if (!(c22 %in% assigned_2022) && !(c17 %in% assigned_2017)) {
    assigned_2022 <- c(assigned_2022, c22)
    assigned_2017 <- c(assigned_2017, c17)
    selected_edges[[length(selected_edges) + 1]] <- r %>% mutate(source = "ambig_greedy")
  }
}
selected_df <- bind_rows(selected_edges)

# fallback try to assign any remaining unassigned 2022 using best candidate with unassigned 2017
unassigned_2022 <- setdiff(unique(cw_pairs$`X2022.NAICS.Code`), selected_df$chosen_2022_code)
if (length(unassigned_2022) > 0) {
  for (c22 in unassigned_2022) {
    cand <- edge_list %>% filter(chosen_2022_code == c22)
    if (nrow(cand) == 0) next
    pick_idx <- which(!(cand$chosen_2017_code %in% assigned_2017))[1]
    if (!is.na(pick_idx)) {
      row <- cand[pick_idx, , drop = FALSE] %>% mutate(source = "ambig_fallback")
      assigned_2022 <- c(assigned_2022, as.character(row$chosen_2022_code))
      assigned_2017 <- c(assigned_2017, as.character(row$chosen_2017_code))
      selected_df <- bind_rows(selected_df, row)
    }
  }
}

final_match <- selected_df %>%
  distinct(chosen_2022_code, chosen_2017_code, .keep_all = TRUE) %>%
  arrange(chosen_2022_code)

# ---- 6. Attach emissions (no aggregation beyond earlier safe collapse if emissions had duplicates) ----
final_with_em <- final_match %>%
  left_join(em_for_join %>% select(`X2017.NAICS.Code`, em_without, em_with),
            by = c("chosen_2017_code" = "X2017.NAICS.Code"))

# If any chosen 2017 codes have no emissions joined, they will have NA em_without/em_with.
# Save final result
write.csv(final_with_em, "final_one_to_one_match_with_emissions.csv", row.names = FALSE)
#note above is the file you want to use in terms of matching 2017 NAICS codes and emissions with the appropriate 2022 NAICS code. 

# Diagnostics: save ambiguous edge list and unmatched pairs
write.csv(edge_list, "ambiguous_edge_list_sorted_by_sim.csv", row.names = FALSE)
chosen_key <- paste0(final_with_em$chosen_2022_code, "||", final_with_em$chosen_2017_code)
all_key <- paste0(cw_pairs$`X2022.NAICS.Code`, "||", cw_pairs$`X2017.NAICS.Code`)
unmatched_idx <- which(!(all_key %in% chosen_key))
if (length(unmatched_idx) > 0) {
  write.csv(cw_pairs[unmatched_idx, ], "unmatched_pairs_after_matching.csv", row.names = FALSE)
} else {
  message("All candidate pairs were either selected or covered by matching.")
}

message("Final one-to-one mapping with emissions saved to 'final_one_to_one_match_with_emissions.csv'.")

#check out the results from above on 2/11/26
#lets check for duplicates
names(final_with_em)
duplicate_rows <- final_with_em%>%
  group_by(chosen_2022_code) %>%
  filter(n() > 1) %>%
  ungroup()
print(duplicate_rows)

dim(final_with_em)
#lets make sure that 1012 unique NAICS codes are in original crosswalkfile..
#remove duplicates from the crosswalk, and then look at dimensions..
names(crosswalk)
crosswalk_no2022dup <- crosswalk%>%
  distinct(X2022.NAICS.Code, .keep_all = TRUE)   
dim(crosswalk_no2022dup)
#hmm.. there is only 1012 in the final_with_em...  that seems odd.. very close but lets find missing one. 
missing_2022 <- setdiff(levels(crosswalk_no2022dup$X2022.NAICS.Code), levels(final_with_em$chosen_2022_code))

print(missing_2022)

#lets check a different way

# 1. Check types & structure
str(crosswalk_no2022dup$X2022.NAICS.Code)
str(final_with_em$chosen_2022_code)

# 2. Convert to plain character vectors and trim whitespace
cw_codes <- trimws(as.character(crosswalk_no2022dup$X2022.NAICS.Code))
fw_codes <- trimws(as.character(final_with_em$chosen_2022_code))

# 3. Count uniques (should show the 1013 vs 1012 discrepancy)
length(unique(cw_codes))
length(unique(fw_codes))

# 4. Check for NA(s)
sum(is.na(cw_codes))
sum(is.na(fw_codes))

# 5. Exact set difference using characters
missing_codes <- setdiff(sort(unique(cw_codes)), sort(unique(fw_codes)))
missing_codes   # this will show code(s) present in crosswalk but not in final_with_em

# 8. Use anti_join (dplyr) to show full rows from crosswalk missing in final
#    Make sure both are character for the join
crosswalk_no2022dup %>%
  mutate(X2022.NAICS.Code = cw_codes) %>%
  anti_join(final_with_em %>% mutate(X2022.NAICS.Code = fw_codes),
            by = "X2022.NAICS.Code")
# we don't need the code for X2022 because its taken care of by the existing 517111 2022 code in the final_with_em dataframe .. so all good. 

# now lets do the opposite check

# 1. Check types & structure
str(crosswalk_no2022dup$X2017.NAICS.Code)
str(final_with_em$chosen_2017_code)

# 2. Convert to plain character vectors and trim whitespace
cw_codes <- trimws(as.character(crosswalk_no2022dup$X2017.NAICS.Code))
fw_codes <- trimws(as.character(final_with_em$chosen_2017_code))

# 3. Count uniques (should show the 1013 vs 1012 discrepancy)
length(unique(cw_codes))
length(unique(fw_codes))

#great!

