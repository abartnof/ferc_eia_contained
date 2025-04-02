# Create the data that the model_generator_id will train on/predict on,
# using all the upstream prev. created helper tables

# TODO: When we were selecting between models, the final value scaling, and 
# replacing NAs with 0.0, was done in python as part of a pipeline. For the final
# version of this script, this will have to be done in this script.

# PER SE
# is_100%: if the plant_name_ferc1 contains a %, indicate if it's at 100%
# eia.plant_part
# eia.technology_description
# eia.prime_mover
# tokens: look for (unit, station, peaker): these keywords often precede numbers

# COMPARISONS
# is_generator_int_referenced: if the plant_name_ferc1 has a non-% integer 
#   (or a sequence of numbers referenced), is the number within generator id referenced?
# capacity_mw_margin (do this in real time)
# string metrics: plant name
# string metrics: utility name

library(tidyverse)
library(skimr)
library(arrow)
library(recipes)

#### Import tables ####

############################################################
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
############################################################

fn_combo <- file.path(data_dir, 'working_data/matches_and_mismatches.parquet')
fn_eia_plant_parts <-file.path(data_dir, 'input_data/eia_plant_parts.RDS')
fn_ferc_steam <-file.path(data_dir, 'input_data/ferc_steam.RDS')
fn_ferc_utility <-file.path(data_dir, 'input_data/ferc_utilities.RDS')

dir_working <- file.path(data_dir, 'working_data/model_b')
fn_ferc1_is_100 <- file.path(dir_working, 'product_record_id_ferc1_to_is_100_percent.parquet')
fn_ferc1_to_numbers <- file.path(dir_working, 'product_record_id_ferc1_to_numbers.RDS')
fn_ferc1_to_token <- file.path(dir_working, 'product_plant_name_ferc1_to_token.parquet')
fn_plant_name_lv_distance <- file.path(dir_working, 'product_plant_name_lv_distance.parquet')
fn_utility_name_lv_distance <- file.path(dir_working, 'product_utility_name_lv_distance.parquet')

dir_tranches <- file.path(data_dir, 'working_data/tranches_ferc_to_eia')

fn_training_x <- file.path(dir_working, '/model_b_training/x.parquet')
fn_training_y <- file.path(dir_working, '/model_b_training/y.parquet')
fn_training_id <- file.path(dir_working, '/model_b_training/id.parquet')
fn_ferc_to_fold <- file.path(data_dir, '/working_data/ferc_to_fold.parquet')


# Combo <- read_parquet(fn_combo)
# FercSteam <- readRDS(fn_ferc_steam)
# FercUtility <- readRDS(fn_ferc_utility)
# EiaPlantParts <- readRDS(fn_eia_plant_parts)
# 
# ProductRecordIdFerc1ToIs100Percent <- read_parquet(fn_ferc1_is_100)
# ProductRecordIdFerc1ToNum <- readRDS(fn_ferc1_to_numbers)
# ProductPlantNameFerc1ToToken <- read_parquet(fn_ferc1_to_token)
# ProductPlantNameLvDistance <- read_parquet(fn_plant_name_lv_distance)
# ProductUtilityNameLvDistance <- read_parquet(fn_utility_name_lv_distance)

#### Define Functions ####

get_ferc_context <- function(FercSteam, FercUtility){
	FercSteam %>%
	left_join(FercUtility, by = 'utility_id_ferc1') %>%
	select(record_id_ferc1, utility_name_ferc1, plant_name_ferc1, capacity_mw) %>%
	rename(capacity_mw_ferc1 = capacity_mw)
}

get_eia_context <- function(EiaPlantParts){
	# Pull the variables we need, making sure to recode tech descriptions and 
	# prime mover codes so that our data can be as simple as possible.
	EiaPlantParts %>%
	mutate(
		generator_integer = parse_integer(str_extract(generator_id, '\\d+')),
	) %>%
	select(
		record_id_eia, 
		utility_name_eia,
		plant_name_eia, 
		# generator_id,
		generator_integer, 
		plant_part, 
		capacity_mw,
		technology_description,
		prime_mover_code
	) %>%
	rename(capacity_mw_eia = capacity_mw) %>%
	mutate(
		technology_description = fct_recode(technology_description,
																				'(Other)' = 'Coal Integrated Gasification Combined Cycle',
																				'(Other)' = 'Solar Thermal with Energy Storage',
																				'(Other)' = 'Natural Gas with Compressed Air Storage',
																				'(Other)' = 'Hydrokinetic',
																				'(Other)' = 'All Other'
		),
		prime_mover_code = fct_recode(prime_mover_code, 
																	'(Other)' = 'OT',
																	'(Other)' = 'CP',
																	'(Other)' = 'CE',
																	'(Other)' = 'HA',
																	'CC' = 'CS',
																	'CS' = 'CA',
																	'CS' = 'CT'
		),
		plant_part = factor(plant_part)
	)
}

#### Join data, perform conduct ad-hoc comparisons ####

get_prepped_combo <- function(
		Combo, 
		FercContext, 
		EiaContext, 
		ProductRecordIdFerc1ToIs100Percent, 
		ProductPlantNameFerc1ToToken,
		ProductPlantNameLvDistance,
		ProductUtilityNameLvDistance
		){
	# Performs all the initial data engineering that doesn't require the 
	#		recipe package.
	# Combo: A tibble with 2 columns, record_id_ferc1, and record_id_eia
		Combo %>%
		# Join FercContext and EiaContext variables
		# Make missing values explicit
		left_join(FercContext, by = 'record_id_ferc1') %>%
		left_join(EiaContext, by = 'record_id_eia') %>%
		mutate(
			technology_description = fct_explicit_na(technology_description),
			prime_mover_code = fct_explicit_na(prime_mover_code),
			plant_part = fct_explicit_na(plant_part),
		) %>%
		left_join(ProductRecordIdFerc1ToNum, by = 'record_id_ferc1') %>%
		# capacity_mw margin
		mutate(
			margin = abs(1 - (capacity_mw_eia / capacity_mw_ferc1)),
			margin = if_else(is.finite(margin), margin, NA),
			margin = round(margin, 3)
		) %>% 
		# is_generator_integer_within_plant_name_ferc1
		mutate(
			is_generator_integer_within_plant_name_ferc1 = 
				!is.na(does_plant_name_ferc1_contain_multiple_numbers) &
				!is.na(generator_integer) &
				map2_lgl(generator_integer, num, ~.x %in% .y)
		) %>%
		# Data per se (ie no comparisons need be done)
		left_join(ProductRecordIdFerc1ToIs100Percent, by = 'record_id_ferc1') %>%
		left_join(ProductPlantNameFerc1ToToken, by = 'plant_name_ferc1') %>%
		mutate_at(vars(starts_with('token__')), replace_na, FALSE)  %>%
		# Comparisons (precreated)
		left_join(ProductPlantNameLvDistance, by = c('plant_name_ferc1', 'plant_name_eia'))  %>%
		left_join(ProductUtilityNameLvDistance, by = c('utility_name_ferc1', 'utility_name_eia')) %>%
		select(
			# record_id_ferc1,
			# record_id_eia,
			# is_match, NB no longer needed
			# utility_name_ferc1,
			# plant_name_ferc1,
			capacity_mw_ferc1,
			capacity_mw_eia,
			margin,
			# utility_name_eia,
			# plant_name_eia,
			# generator_integer,
			plant_part,
			technology_description,
			prime_mover_code,
			# num,
			does_plant_name_ferc1_contain_multiple_numbers,
			is_generator_integer_within_plant_name_ferc1,
			is_100_percent,
			token__cc,
			token__com,
			token__comb,
			token__combined,
			token__cycle,
			token__diesel,
			token__gas,
			token__gt,
			token__nuclear,
			token__number,
			token__peaker,
			token__share,
			token__st,
			token__station,
			token__steam,
			token__total,
			token__turbine,
			token__unit,
			token__wind,
			plant_name_dist_lv,
			ratio_lv_to_plant_name_ferc1_len,
			utility_name_dist_lv,
			ratio_lv_to_utility_name_ferc1_len
			)
}

get_recipe_fit_training <- function(PreppedCombo){
	recipe( ~ ., data = PreppedCombo) %>%
	# Factors
	step_dummy(all_factor_predictors(), one_hot = TRUE) %>%
	# Logicals
	step_mutate_at(all_logical_predictors(), fn = ~as.integer(.)) %>%
	# Scale
	# step_normalize(all_predictors()) %>%
	# step_mutate_at(all_predictors(), fn = ~replace_na(., 0.0)) %>%
	prep
}

get_recipe_fit_tranches <- function(PreppedCombo){
	recipe( ~ ., data = PreppedCombo) %>%
	# step_zv() %>%
	# Factors
	step_dummy(all_factor_predictors(), one_hot = TRUE) %>%
	# Logicals
	step_mutate_at(all_logical_predictors(), fn = ~as.integer(.)) %>%
	# Scale
	step_normalize(all_predictors()) %>%
	step_mutate_at(all_predictors(), fn = ~replace_na(., 0.0)) %>%
	prep
}
