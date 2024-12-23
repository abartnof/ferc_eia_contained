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
# 	FOR CURIOUS HUMANS TO LOOK AT:
# 	train_id.parquet
# 	validation_id.parquet
# 	test_id.parquet
#		train_y.parquet
#		validation_y.parquet

#		FOR MODELS:
#		test_y.parquet
# 	train_x.parquet
# 	test_x.parquet
# 	validation_x.parquet
#-------------------------------------------------------------------------------

library(tidyverse)
library(recipes)
library(rsample)
library(arrow)
set.seed(1)


# dir_input <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/input_data/'
dir_working  <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/'
fn_all_joined_data <- file.path(dir_working, 'all_joined_data.parquet')
JoinedData <- read_parquet(fn_all_joined_data)

dir_train <- file.path(dir_working, 'model_a/train/')
dir.create(dir_train, showWarnings = TRUE)
# X
fn_train_x			<- file.path(dir_train, 'train_x.parquet')
fn_test_x 			<- file.path(dir_train, 'test_x.parquet')
# fn_validation_x <- file.path(dir_train, 'validation_x.parquet')
# y
fn_train_y			<- file.path(dir_train, 'train_y.parquet')
fn_test_y 			<- file.path(dir_train, 'test_y.parquet')
# fn_validation_y <- file.path(dir_train, 'validation_y.parquet')
# ID (eg interesting data that won't be used for the model)
fn_train_id			 <- file.path(dir_train, 'train_id.parquet')
fn_test_id 			 <- file.path(dir_train, 'test_id.parquet')
# fn_validation_id <- file.path(dir_train, 'validation_id.parquet')

#### Split data ####

# We want the training set to have one true match, and 2k false mappings, 
# for each ferc id; likewise for the test set. 
# this means we must sample not by row, but by ferc id
unique_ferc_ids <- unique(JoinedData$record_id_ferc1)
num_ferc_ids_in_training_set <- round(
	0.8 * length(unique_ferc_ids)
)
training_ferc_ids <- sample(x = unique_ferc_ids, size = num_ferc_ids_in_training_set, replace = FALSE)

RecordIdFerc1ForTrain <-
	training_ferc_ids %>%
	enframe(name=NULL, value='record_id_ferc1') %>%
	mutate(set = 'train')

RecordIdFerc1ForTest <-
	setdiff(unique_ferc_ids, training_ferc_ids) %>%
	enframe(name=NULL, value='record_id_ferc1') %>%
	mutate(set = 'test')
#
Train <- 
	JoinedData %>% 
	semi_join(RecordIdFerc1ForTrain, by = 'record_id_ferc1')

Test <-
	JoinedData %>% 
	semi_join(RecordIdFerc1ForTest, by = 'record_id_ferc1')

#### Export non-predictor variables ####
Train %>%
	select(record_id_eia, record_id_ferc1, report_year) %>%
	write_parquet(fn_train_id) # write_parquet('train_id.parquet')
# Validation %>%
# 	select(record_id_eia, record_id_ferc1, report_year) %>%
# 	write_parquet(fn_validation_id) # write_parquet('validation_id.parquet')
Test %>%
	select(record_id_eia, record_id_ferc1, report_year) %>%
	write_parquet(fn_test_id) # write_parquet('test_id.parquet')

Train %>%
	select(is_match) %>%
	mutate(is_match = as.integer(is_match)) %>%
	write_parquet(fn_train_y) # write_parquet('train_y.parquet')
# Validation %>%
# 	select(is_match) %>%
# 	mutate(is_match = as.integer(is_match)) %>%
# 	write_parquet(fn_validation_y) # write_parquet('validation_y.parquet')
Test %>%
	select(is_match) %>%
	mutate(is_match = as.integer(is_match)) %>%
	write_parquet(fn_test_y) # write_parquet('test_y.parquet')

TrainPred <- 
	Train %>%
	select(-record_id_ferc1, -record_id_eia, -report_year, -is_match)
TestPred <- 
	Test %>%
	select(-record_id_ferc1, -record_id_eia, -report_year, -is_match)
# ValidationPred <- 
# 	Test %>%
# 	select(-record_id_ferc1, -record_id_eia, -report_year, -is_match)

#### Process factors ####

good_variables_plant_type_ferc1 <- c('steam', 'combustion_turbine',
	'combined_cycle', 'nuclear', 'internal_combustion', 'photovoltaic')
good_variables_technology_description_eia <- c('Petroleum Liquids',
	'Natural Gas Fired Combined Cycle', 'Natural Gas Fired Combustion Turbine',
	'Conventional Steam Coal', 'Landfill Gas',
	'Natural Gas Internal Combustion Engine', 'Natural Gas Steam Turbine')
good_variables_prime_mover_code_eia <- c('IC', 'ST', 'GT', 'CT', 'CA')

collapse_factors <- function(X){
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
TrainPred <- collapse_factors(TrainPred)
# ValidationPred <- collapse_factors(ValidationPred)
TestPred <- collapse_factors(TestPred)

#### Predictor processing ####
recipe_fit <-
	recipe( ~ ., data = TrainPred) %>%
	
	# Factors
	step_dummy(all_factor_predictors(), one_hot = TRUE) %>%

	# Logicals
	step_mutate_at(all_logical_predictors(), fn = ~as.integer(.)) %>%
	
	# Scale all predictors
	step_zv(all_predictors()) %>%
	step_center(all_predictors(), na_rm = TRUE) %>%
	step_scale(all_predictors(), na_rm = TRUE) %>%
	
	# Range
	step_range(all_predictors(), min = -3.0, max = 3.0) %>%
	
	# Missing numeric values
	step_mutate_at(all_numeric_predictors(), fn = ~replace_na(., 0.0)) %>%
	prep
print(recipe_fit)

bake(recipe_fit, TrainPred) %>%
	write_parquet(fn_train_x) # 	write_parquet('train_x.parquet')
TestFit <- bake(recipe_fit, TestPred) %>%
	write_parquet(fn_test_x) # write_parquet('test_x.parquet')
# ValidationFit <- bake(recipe_fit, ValidationPred) %>%
# 	write_parquet(fn_validation_x) # write_parquet('validation_x.parquet')
