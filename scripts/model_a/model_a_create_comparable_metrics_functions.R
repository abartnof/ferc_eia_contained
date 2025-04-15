# Establish functions to be sourced when comparing easily-comparable metrics
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof

library(arrow)
library(dtplyr)
library(stringdist)
library(tidyverse)

#### Define functions ####
get_comparable_scalars <- function(Combos, FercSteam, EiaPlantParts){
	# Arguments
		# Combos: A table that contains both record_id_ferc1, and record_id_eia. For training, this will be the Combos table. Otherwise, it will be a Tranche file.
		# FercSteam: The full FercSteam table
		# EiaPlantParts: The full Eia Plant Parts table
	# Returns: ComparableScalars, a table of the comparable scalars wrt the FERC and EIA records
	# Note: all of the input tables are expected to be lazy_dt tables, NOT tibbles.
	# Compares the following variables:
		# "construction_year" 
		# "installation_year"  
		# "capacity_mw"        
		# "net_generation_mwh"
	# Ignores report_year, because we already block on this.
	
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
	return(ComparableScalars)
}

get_stopwords_plantname <- function(EiaPlantParts){
	# Arguments:
		# EiaPlantParts: The full EIA Plant Parts table
	# Returns: StopWordsPlantName, a table that contains all of the tokens we'll exclude

	# Create plant name tokens	
	PlantTokens <-
		EiaPlantParts %>%
		select(plant_name_eia) %>%
		as_tibble %>%
		mutate(
			token = str_split(plant_name_eia, '\\b')
		)
	
	# Note the prevalance of each token
	PlantECDF <-
		PlantTokens %>%
		unnest(token) %>%
		mutate(token = str_replace_all(token, '\\[|\\]|\\(|\\)', '') ) %>%
		filter(str_length(token) > 1L) %>%
		count(token) %>%
		arrange(n) %>%
		mutate(prob = ecdf(n)(n))

	# Classify the most common tokens as stopwords	
	StopWordsPlantName <-
		PlantECDF %>%
		filter(prob > 0.99) %>%
		select(token)
	return(StopWordsPlantName)
}

get_stopwords_utility_name <- function(EiaPlantParts){
	# Arguments:
		# EiaPlantParts: The full EIA Plant Parts table
	# Returns: StopWordsUtilityName, a table that contains all of the tokens we'll exclude
	
	# Create utility name tokens	
	UtilityTokens <-
		EiaPlantParts %>%
		select(utility_name_eia) %>%
		as_tibble %>%
		mutate(
			token = str_split(utility_name_eia, '\\b')
		)
	
	# Note the prevalance of each token
	UtilityECDF <-
		UtilityTokens %>%
		unnest(token) %>%
		mutate(token = str_replace_all(token, '\\[|\\]\\(\\)', '')) %>%
		filter(str_length(token) > 1L) %>%
		count(token) %>%
		arrange(n) %>%
		mutate(prob = ecdf(n)(n))
	
	# Classify the most common tokens as stopwords	
	StopWordsUtilityName <-
		UtilityECDF %>%
		filter(prob > 0.99) %>%
		select(token)
	return(StopWordsUtilityName)
}

get_regex_uber_pattern <- function(StopWords){
	# Concatenate all stopwords into one massive regex pattern, 
	# separated by the vertical pipe (ie 'or' in regex'). 
	# eg, ('foo', 'bar') becomes 'foo|bar'
	StopWords %>%
	mutate(
		token_adj = str_c('\\b', token, '\\b')
	) %>%
	pull(token_adj) %>%
	paste(., collapse = '|')
}

get_string_metrics <- function(Combos, EiaPlantParts, FercSteam, FercUtilities, stop_words_plant_name_uber_pattern, stop_words_utility_name_uber_pattern){
	# Arguments:
		# Combos: Combos (training) or tranche (predicting all data)
		# EiaPlantParts
		# FercSteam
		# FercUtilities
		# stop_words_plant_name_uber_pattern
		# stop_words_utility_name_uber_pattern
	# Returns:
		# All kinds of string distance: plant name | utility name, original EIA | sans stopwords
	
	# Get EIA Names, stripped of stopwords
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
	
	# Get FERC strings, as-is
	FercNames <-
		Combos %>%
		select(record_id_ferc1) %>%
		left_join(FercSteam, by = 'record_id_ferc1') %>%
		left_join(FercUtilities, by = 'utility_id_ferc1') %>%
		select(plant_name_ferc1, utility_name_ferc1) %>%
		collect
	
	# calculate string metrics between EIA names (refined or original) and FERC names
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
	
	return(StringDistance)
}

get_additional_columns <- function(Combos, FercSteam, EiaPlantParts){
	# Collect additional data-- neither scalar nor strings, 
	# just data that are entered into the models per se
	# Arguments: 
		# Combos: Combos (training) or tranche (predicting all data)
		# FercSteam
		# EiaPlantParts
	# Returns: The additional columns
	
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
	
	return( bind_cols(AdditionalFercColumns, AdditionalEiaColumns) )
}

#### Run scripts ####
# ComparableScalars <- get_comparable_scalars(Combos = Combos, FercSteam = FercSteam, EiaPlantParts = EiaPlantParts)
# StopWordsPlantName <- get_stopwords_plantname(EiaPlantParts = EiaPlantParts)
# StopWordsUtilityName <- get_stopwords_utility_name(EiaPlantParts = EiaPlantParts)
# 
# # Run if training, as note-keeping:
# StopWordsPlantName %>%
# 	write_csv(fn_stop_words_plant_name)
# StopWordsUtilityName %>%
# 	write_csv(fn_stop_words_utility_name)
# 
# stop_words_plant_name_uber_pattern <- get_regex_uber_pattern(StopWords = StopWordsPlantName)
# stop_words_utility_name_uber_pattern <- get_regex_uber_pattern(StopWords = StopWordsUtilityName)
# 
# StringMetrics <- get_string_metrics(
# 	Combos = Combos, 
# 	EiaPlantParts = EiaPlantParts, 
# 	FercSteam = FercSteam, 
# 	FercUtilities = FercUtilities, 
# 	stop_words_plant_name_uber_pattern = stop_words_plant_name_uber_pattern, 
# 	stop_words_utility_name_uber_pattern = stop_words_utility_name_uber_pattern
# )
# 
# AdditionalColumns <- get_additional_columns(
# 	Combos = Combos, 
# 	FercSteam = FercSteam, 
# 	EiaPlantParts = EiaPlantParts
# )
# 
# Output <-
# 	Combos %>%
# 	collect %>%
# 	bind_cols(StringMetrics) %>%
# 	bind_cols(ComparableScalars) %>%
# 	bind_cols(AdditionalColumns)
# 
# 	# write_parquet(fn_all_joined_data)
# 
# 
# 
# 
# 
