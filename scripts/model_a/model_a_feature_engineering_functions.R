#-------------------------------------------------------------------------------
# feature_engineering_hyperparameter_search.R
# 
# Within the full neural network pipeline, 
# prepare the training/testing datasets necessary for a hyperparameter search.
# 
# Author: Andrew Bartnof, for RMI
# Email: abartnof.contractor@rmi.org
# 2024
# 
# input: 
# 	all_joined_data.parquet
# output: 

#		FOR MODELS:
#		y.parquet
# 	x.parquet

# 	validation_x.parquet
#-------------------------------------------------------------------------------

library(tidyverse)
library(recipes)
library(rsample)
library(arrow)
set.seed(1)

# data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
# dir_input <- file.path(data_dir, '/input_data/')
# dir_working <- file.path(data_dir, '/working_data/')
# dir_working_model_a_training <- file.path(
# 	data_dir, 
# 	'/working_data/model_a/model_a_training/'
# )
# 
# fn_all_joined_data <- file.path(dir_working_model_a_training, 'all_joined_data.parquet')
# fn_id <- file.path(dir_working_model_a_training, 'id.parquet')
# fn_y <- file.path(dir_working_model_a_training, 'y.parquet')
# fn_x <- file.path(dir_working_model_a_training, 'x.parquet')
# 
# JoinedData <- read_parquet(fn_all_joined_data)


#### Training data only: assign a fold num to each record_id_ferc1 ####
get_ferc_to_fold <- function(JoinedData){
	unique_ferc_ids <- unique(JoinedData$record_id_ferc1)
	fold_range <- c(0L, 1L, 2L, 3L, 4L)
	fold_vector <- sample(x = fold_range, size = length(unique_ferc_ids), replace = TRUE)
	FercToFold <-
		tibble(record_id_ferc1 = unique_ferc_ids, fold_num = fold_vector)
	return(FercToFold)
}

#### Export non-predictor variables ####
get_id <- function(JoinedData){
	JoinedData %>%
	select(record_id_ferc1, record_id_eia, report_year)
}

get_y <- function(JoinedData){
	JoinedData %>%
	select(is_match) %>%
	mutate(is_match = as.integer(is_match))
}

get_predictors <- function(JoinedData){
	JoinedData %>%
	select(-record_id_ferc1, -record_id_eia, -report_year, -is_match)
}

collapse_factors <- function(X){
	
	good_variables_plant_type_ferc1 <- c('steam', 'combustion_turbine',
		'combined_cycle', 'nuclear', 'internal_combustion', 'photovoltaic')
	good_variables_technology_description_eia <- c('Petroleum Liquids',
		'Natural Gas Fired Combined Cycle', 'Natural Gas Fired Combustion Turbine',
		'Conventional Steam Coal', 'Landfill Gas',
		'Natural Gas Internal Combustion Engine', 'Natural Gas Steam Turbine')
	good_variables_prime_mover_code_eia <- c('IC', 'ST', 'GT', 'CT', 'CA')
	
	X %>%
	mutate(
		plant_type_ferc1 = fct_other(plant_type_ferc1, 
																 keep = good_variables_plant_type_ferc1),
		technology_description_eia = fct_other(technology_description_eia, 
																 keep = good_variables_technology_description_eia),
		prime_mover_code_eia = fct_other(prime_mover_code_eia, 
																 keep = good_variables_prime_mover_code_eia)
	) %>%
	mutate_if(is.factor, factor, ordered = FALSE) %>%
	mutate_if(is.factor, fct_explicit_na)
}

# The training version of this is NOT scaled. This is so that we can 
# do this in the cross-validation step. OTOH, once we're working with
# tranches of data, we already have a fit model, and we want to scale
# based on our training dataset.
get_recipe_fit_training <- function(PredictorsFactored){
	recipe( ~ ., data = PredictorsFactored) %>%
	step_dummy(all_factor_predictors(), one_hot = TRUE) %>%
	step_mutate_at(all_logical_predictors(), fn = ~as.integer(.)) %>%
	prep
}
get_recipe_fit_tranches <- function(PredictorsFactored){
	recipe( ~ ., data = PredictorsFactored) %>%
	step_dummy(all_factor_predictors(), one_hot = TRUE) %>%
	step_mutate_at(all_logical_predictors(), fn = ~as.integer(.)) %>%
		
	# Scale all predictors
	# step_zv(all_predictors()) %>%
	step_center(all_predictors(), na_rm = TRUE) %>%
	step_scale(all_predictors(), na_rm = TRUE) %>%

	# Range
	step_range(all_predictors(), min = -3.0, max = 3.0) %>%
	# 
	# Missing numeric values
	step_mutate_at(all_numeric_predictors(), fn = ~replace_na(., 0.0)) %>%
	prep
}


# # Script (Training Data version)
# get_id(JoinedData) %>%
# left_join(FercToFold, by = 'record_id_ferc1') %>%
# write_parquet(fn_id)
# 
# get_y(JoinedData) %>%
# 	write_parquet(fn_y)
# 
# Predictors <- get_predictors(JoinedData)
# PredictorsFactored <- collapse_factors(Predictors)
# recipe_fit <- get_recipe_fit(PredictorsFactored)
# 
# bake(recipe_fit, PredictorsFactored) %>%
# 	write_parquet(fn_x)
