#-------------------------------------------------------------------------------
# model_a_train__create_training_data_comparable_metrics.R

# For any EIA/FERC metrics that should be compared,
# (eg we want the ratio of two numbers, the difference, etc), 
# pre-compute them here.
# For any strings that should be compared, compare them two ways:
# as the full strings, and with stopwords (established and) removed

# input: 
	# ferc_steam.RDS
	# ferc_utilities.RDS
	# eia_plant_parts.RDS
	# matches_and_mismatches.parquet
# output:
	# stop_words_plant_name.csv
	# stop_words_utility.csv
	# all_joined_data.parquet

# Author: Andrew Bartnof, for RMI
# Email: abartnof.contractor@rmi.org
# 2024

#-------------------------------------------------------------------------------

library(arrow)
library(dtplyr)
library(skimr)
library(stringdist)
library(tidyverse)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_input <- file.path(data_dir, '/input_data/')
dir_working <- file.path(data_dir, '/working_data/')
dir_working_model_a_training <- file.path(
	data_dir, 
	'/working_data/model_a/model_a_training/'
)

fn_eia_plant_parts <- file.path(dir_input, 'eia_plant_parts.RDS')
fn_ferc_steam <- file.path(dir_input, 'ferc_steam.RDS')
fn_ferc_utilities <- file.path(dir_input, 'ferc_utilities.RDS')
fn_matches_and_mismatches <- file.path(dir_working, 'matches_and_mismatches.parquet')

fn_all_joined_data <- file.path(dir_working_model_a_training, 'all_joined_data.parquet')
fn_stop_words_plant_name <- file.path(dir_working_model_a_training, 'stop_words_plant_name.csv')
fn_stop_words_utility_name <- file.path(dir_working_model_a_training, 'stop_words_utility.csv')

FercSteam <- readRDS(fn_ferc_steam) %>% lazy_dt
FercUtilities <- readRDS(fn_ferc_utilities) %>% lazy_dt
EiaPlantParts <- readRDS(fn_eia_plant_parts) %>% lazy_dt
Combos <- read_parquet(fn_matches_and_mismatches) %>% lazy_dt

#### Numeric/scalar predictors ####
# columns to use:
	# "construction_year" 
	# "installation_year"  
	# "capacity_mw"        
	# "net_generation_mwh"
# columns to ignore:
	# "report_year" (don't use, we already block on this)

FercComparableScalars <-
	Combos %>%
	select(record_id_ferc1) %>%
	left_join(FercSteam, by = 'record_id_ferc1') %>%
	select(construction_year, installation_year, capacity_mw, net_generation_mwh) %>%
	rename_all(~str_c(., '_ferc1')) %>%
	collect

EiaComparableScalars <-
	Combos %>%
	select(record_id_eia) %>%
	left_join(EiaPlantParts, by = 'record_id_eia') %>%
	select(construction_year, installation_year, capacity_mw, net_generation_mwh) %>%
	rename_all(~str_c(., '_eia')) %>%
	collect

ComparableScalars <-
	FercComparableScalars %>%
	bind_cols(EiaComparableScalars) %>%
	mutate(
		construction_year_ratio = construction_year_ferc1 / construction_year_eia,
		installation_year_ratio = installation_year_ferc1 / installation_year_eia,
		capacity_mw_ratio = capacity_mw_ferc1 / capacity_mw_eia,
		net_generation_mwh_ratio = net_generation_mwh_ferc1 / net_generation_mwh_eia
	) %>%
	mutate_at(vars(contains('ratio')), na_if, Inf) %>%
	mutate_at(vars(contains('ratio')), na_if, -Inf)

#### Strings ####
# Note strings two ways: 
	# the full strings, 
	# and the strings with stopwords removed.

#### Plant Names ####
# Convert strings to tokens
PlantTokens <-
	EiaPlantParts %>%
	select(plant_name_eia) %>%
	as_tibble %>%
	mutate(
		token = str_split(plant_name_eia, '\\b')
	)

UtilityTokens <-
	EiaPlantParts %>%
	select(utility_name_eia) %>%
	as_tibble %>%
	mutate(
		token = str_split(utility_name_eia, '\\b')
	)

# Find stopwords
PlantECDF <-
	PlantTokens %>%
	unnest(token) %>%
	mutate(token = str_replace_all(token, '\\[|\\]|\\(|\\)', '') ) %>%
	filter(str_length(token) > 1L) %>%
	count(token) %>%
	arrange(n) %>%
	mutate(prob = ecdf(n)(n))

StopWordsPlantName <-
	PlantECDF %>%
	filter(prob > 0.99) %>%
	select(token)

nrow(StopWordsPlantName)
print('Plant name stop words:')
print(StopWordsPlantName$token)

StopWordsPlantName %>%
	write_csv(fn_stop_words_plant_name)

UtilityECDF <-
	UtilityTokens %>%
	unnest(token) %>%
	mutate(token = str_replace_all(token, '\\[|\\]\\(\\)', '')) %>%
	filter(str_length(token) > 1L) %>%
	count(token) %>%
	arrange(n) %>%
	mutate(prob = ecdf(n)(n))

StopWordsUtilityName <-
	UtilityECDF %>%
	filter(prob > 0.99) %>%
	select(token)

nrow(StopWordsUtilityName)
print('Utility stop words:')
print(StopWordsUtilityName$token)

StopWordsUtilityName %>%
	write_csv(fn_stop_words_utility_name)

# Create a single regex pattern that represents all of the stopwords, separated
# by the vertical pipe (ie 'or' in regex'). 
# eg, ('foo', 'bar') becomes 'foo|bar'
stop_words_plant_name_uber_pattern <-
	StopWordsPlantName %>%
	mutate(
		token_adj = str_c('\\b', token, '\\b')
	) %>%
	pull(token_adj) %>%
	paste(., collapse = '|')
# print(stop_words_plant_name_uber_pattern)

stop_words_utility_name_uber_pattern <-
	StopWordsUtilityName %>%
	mutate(
		token_adj = str_c('\\b', token, '\\b')
	) %>%
	pull(token_adj) %>%
	paste(., collapse = '|')
# print(stop_words_utility_name_uber_pattern)

# Collect eia plant names, utility names, and then strip stopwords from them
EiaNames <-
	Combos %>%
	left_join(EiaPlantParts, by = 'record_id_eia') %>% 
	select(plant_name_eia, utility_name_eia) %>%
	mutate(
		plant_name_eia_refined = str_replace_all(
			plant_name_eia,
			stop_words_plant_name_uber_pattern,
			' '),
		utility_name_eia_refined = str_replace_all(
			utility_name_eia,
			stop_words_utility_name_uber_pattern,
			' ')
	) %>%
	mutate_at(vars(contains('refined')), ~str_replace_all(., '[:punct:]+', ' ')) %>%
	mutate_at(vars(contains('refined')), str_trim) %>%
	mutate_at(vars(contains('refined')), ~str_replace_all(., '[\\s]+', ' ')) %>%
	collect
	

# Get FERC strings
FercNames <-
	Combos %>%
	select(record_id_ferc1) %>%
	left_join(FercSteam, by = 'record_id_ferc1') %>%
	left_join(FercUtilities, by = 'utility_id_ferc1') %>%
	select(plant_name_ferc1, utility_name_ferc1) %>%
	collect

# calculate string metrics
StringDistance <-
	FercNames %>%
	bind_cols(EiaNames) %>%
	mutate(
		dist_plant_name_lv  = stringdist(plant_name_ferc1, plant_name_eia, method = 'lv'),
		dist_plant_name_lcs = stringdist(plant_name_ferc1, plant_name_eia, method = 'lcs'),
		dist_utility_name_lv  = stringdist(utility_name_ferc1, utility_name_eia, method = 'lv'),
		dist_utility_name_lcs = stringdist(utility_name_ferc1, utility_name_eia, method = 'lcs'),
		
		dist_plant_name_refined_lv  = stringdist(plant_name_ferc1, plant_name_eia_refined, method = 'lv'),
		dist_plant_name_refined_lcs = stringdist(plant_name_ferc1, plant_name_eia_refined, method = 'lcs'),
		dist_utility_name_refined_lv  = stringdist(utility_name_ferc1, utility_name_eia_refined, method = 'lv'),
		dist_utility_name_refined_lcs = stringdist(utility_name_ferc1, utility_name_eia_refined, method = 'lcs'),
		
		does_plant_name_ferc_contain_digits = str_detect(plant_name_ferc1, '\\d'),
		does_plant_name_eia_contain_digits  = str_detect(plant_name_eia, '\\d'),
		do_both_plant_names_contain_digits = does_plant_name_ferc_contain_digits & does_plant_name_eia_contain_digits,
		
		does_utility_name_ferc_contain_digits = str_detect(utility_name_ferc1, '\\d'),
		does_utility_name_eia_contain_digits  = str_detect(utility_name_eia, '\\d'),
		do_both_utility_names_contain_digits = does_utility_name_ferc_contain_digits & does_utility_name_eia_contain_digits
	) %>%
	mutate_if(is.logical, as.integer) %>%
	select_if(is.numeric)

# Collect additional data-- neither scalar nor strings
AdditionalFercColumns <-
	Combos %>%
	select(record_id_ferc1) %>%
	left_join(FercSteam, by = 'record_id_ferc1') %>%
	select(plant_type) %>%
	rename(plant_type_ferc1 = plant_type) %>%
	mutate(plant_type_ferc1 = as.factor(plant_type_ferc1)) %>%
	collect

AdditionalEiaColumns <-
	Combos %>%
	select(record_id_eia) %>%
	left_join(EiaPlantParts, by = 'record_id_eia') %>%
	select(technology_description, prime_mover_code, plant_part) %>%
	rename_all(~str_c(., '_eia')) %>%
	mutate_all(as.factor) %>%
	collect

# Write to disk
Combos %>%
	collect %>%
	bind_cols(StringDistance) %>%
	bind_cols(ComparableScalars) %>%
	bind_cols(AdditionalFercColumns) %>%
	bind_cols(AdditionalEiaColumns) %>%
	write_parquet(fn_all_joined_data)

# Combos %>%
# 	count(record_id_ferc1) %>%
# 	filter(n == 1)