# Feature engineering
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof

# Use these locations to make the script work on your machine
dir_scripts <- '~/Documents/rmi/rematch_ferc_eia_pixi/ferc_eia/scripts/'
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
# -----------------------------------------------------------

# Source script
fn_script <- file.path(dir_scripts, 'model_a/model_a_feature_engineering_functions.R')
source(fn_script)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_input <- file.path(data_dir, '/input_data/')
dir_working <- file.path(data_dir, '/working_data/')
dir_working_model_a_training <- file.path(
	data_dir, 
	'/working_data/model_a/model_a_training/'
)

fn_all_joined_data <- file.path(dir_working_model_a_training, 'all_joined_data.parquet')
fn_id <- file.path(dir_working_model_a_training, 'id.parquet')
fn_y <- file.path(dir_working_model_a_training, 'y.parquet')
fn_x <- file.path(dir_working_model_a_training, 'x.parquet')
fn_ferc_to_fold <- file.path(dir_working, 'ferc_to_fold.parquet')


JoinedData <- read_parquet(fn_all_joined_data)

#### Script (Training Data version) ####
# Training data only: assign a fold num to each record_id_ferc1 #
FercToFold <- read_parquet(fn_ferc_to_fold)

# Export non-predictor variables
get_id(JoinedData) %>%
left_join(FercToFold, by = 'record_id_ferc1') %>%
write_parquet(fn_id)

get_y(JoinedData) %>%
	write_parquet(fn_y)

# Export feature-engineered predictor variables
Predictors <- get_predictors(JoinedData)
PredictorsFactored <- collapse_factors(Predictors)
recipe_fit <- get_recipe_fit_training(PredictorsFactored)

bake(recipe_fit, PredictorsFactored) %>%
	write_parquet(fn_x)
