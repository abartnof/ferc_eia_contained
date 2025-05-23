# First step in creating training dataset: 
# for each ‘ground-truth’ mapping, note several thousand non-valid mappings
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof

library(tidyverse)
library(dtplyr)
library(arrow)
set.seed(1)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_input <- file.path(data_dir, '/input_data/')
dir_working <- file.path(data_dir, '/working_data/')

fn_eia_plant_parts <- file.path(dir_input, 'eia_plant_parts.RDS')
fn_ferc_steam <- file.path(dir_input, 'ferc_steam.RDS')
fn_positive_matches <- file.path(dir_working, 'positive_matches.RDS')
fn_matches_and_mismatches <- file.path(dir_working, 'matches_and_mismatches.parquet')

fn_ferc_to_fold <- file.path(dir_working, 'ferc_to_fold.parquet')

NUM_COMPARISONS <- 1000L

EiaPlantParts <- readRDS(fn_eia_plant_parts) %>%
	lazy_dt() 
FercSteam <- readRDS(fn_ferc_steam) %>%
	lazy_dt()
PositiveMatches <- readRDS(fn_positive_matches) %>%
	lazy_dt()

#### Cartesian Products ####
# Note all of the years that we'll iterate through, to make training data.
# We'll exclude 2004-- too few entries

# PositiveMatches %>%
# 	count(report_year) %>%
# 	collect

relevant_year_list <-
	PositiveMatches %>%
	distinct(report_year) %>%
	filter(report_year >= 2005L) %>%
	arrange(report_year) %>%
	collect %>%
	pull(report_year)

# sanity check-- all years we'll be using exist in the EIA and Steam tables?

# all(relevant_year_list %in% unique(collect(EiaPlantParts)$report_year))
# all(relevant_year_list %in% unique(collect(FercSteam)$report_year))

# Iterate through all years, and find records to compare that are NOT matches
RandomCombos <- tibble()
for (relevant_year in relevant_year_list){
	print(relevant_year)
	
	FercColumn <- PositiveMatches %>%
		filter(report_year == relevant_year) %>%
		select(record_id_ferc1) %>%
		collect
	
	EiaColumn <-
		EiaPlantParts %>%
		filter(report_year == relevant_year) %>%
		select(record_id_eia) %>%
		collect
	
	YearlyRandomCombos <-
		FercColumn %>%
		cross_join(EiaColumn) %>%
		anti_join(
			as_tibble(PositiveMatches), by = c('record_id_ferc1', 'record_id_eia')
		) %>%
		group_by(record_id_ferc1) %>%
		slice_sample(n = NUM_COMPARISONS, replace = FALSE) %>%
		ungroup %>%
		mutate(report_year = relevant_year, is_match = FALSE)
	RandomCombos <- RandomCombos %>% bind_rows(YearlyRandomCombos)
}

# Ensure each year has enough data
RandomCombos %>%
	count(report_year, record_id_ferc1, name='false_matches') %>%
	count(report_year, false_matches)

# Make sure: this should be empty, there should be no overlap between
# the training data and the prev. collected comparisons

# RandomCombos %>%
# 	lazy_dt() %>%
# 	semi_join(PositiveMatches) %>%
# 	collect %>%
# 	nrow

# Join matches with mismatches, and write to disk! 
PositiveMatches %>%
	collect %>%
	semi_join(RandomCombos, by = 'record_id_ferc1') %>%  # Only add 'true' mappings if we chose to use this year for false mappings
	mutate(is_match = TRUE) %>%
	bind_rows(RandomCombos) %>%
	arrange(report_year, record_id_ferc1, is_match) %>%
	write_parquet(fn_matches_and_mismatches)


# Create a consistent mapping, 
# linking each FERC record id to a fold 

DistinctRecordIDFerc1 <-
	FercSteam %>%
		distinct(record_id_ferc1) %>%
		collect

fold_vector <- sample(
	x = seq(0, 4), 
	replace = TRUE, 
	size = nrow(DistinctRecordIDFerc1)
)

DistinctRecordIDFerc1 %>%
	bind_cols(fold = fold_vector) %>%
	write_parquet(fn_ferc_to_fold)