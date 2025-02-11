# Run the functions to create the feature engineering model data, as sourced
# from another file
# version: training data

# Use these locations to make the script work on your machine
dir_scripts <- '~/Documents/rmi/rematch_ferc_eia_pixi/ferc_eia/scripts/'
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
# -----------------------------------------------------------

# Source script
fn_script <- file.path(dir_scripts, 'model_a/model_a_feature_engineering_functions.R')
source(fn_script)

# Point to other files
# dir_input <- file.path(data_dir, '/input_data/')
dir_working <- file.path(data_dir, '/working_data/')
dir_working_model_a_training <- file.path(
	data_dir, 
	'/working_data/model_a/model_a_training/'
)

# Locations of tranche-like data files
dir_comparable_metrics <- file.path(dir_working, 'model_a/model_a_comparable_metrics')
dir_x = file.path(dir_working, '/model_a/model_a_x')
# dir_y = file.path(dir_working, '/model_a/model_a_y')
dir_id = file.path(dir_working, '/model_a/model_a_id')

# Load JoinedData-- necessary to fit the recipe
fn_all_joined_data <- file.path(dir_working_model_a_training, 'all_joined_data.parquet')
JoinedData <- read_parquet(fn_all_joined_data)

#### Script (Tranche data version) ####

# Export non-predictor variables
# get_id(JoinedData) %>%
# left_join(FercToFold, by = 'record_id_ferc1') %>%
# write_parquet(fn_id)
# 
# get_y(JoinedData) %>%
# 	write_parquet(fn_y)

# Get all data characteristics from training data for the 'recipe'
Predictors <- get_predictors(JoinedData)
PredictorsFactored <- collapse_factors(Predictors)
recipe_fit <- get_recipe_fit_tranches(PredictorsFactored)
print(recipe_fit)

FN <-
	list.files(dir_comparable_metrics, full.names = FALSE) %>%
	enframe(name = 'i', value = 'fn_comparable_metrics') %>%
	mutate(
		fn_x = str_replace(fn_comparable_metrics, 'comparable_metrics', 'x'),
		fn_x = file.path(dir_x, fn_x),
		fn_id = str_replace(fn_comparable_metrics, 'comparable_metrics', 'id'),
		fn_id = file.path(dir_id, fn_id),
		fn_comparable_metrics = file.path(dir_comparable_metrics, fn_comparable_metrics)
	)

for (i in FN$i){
	print(i)
	TrancheJoinedData <- read_parquet(FN$fn_comparable_metrics[i])
	
	TrancheJoinedData %>%
		select(record_id_ferc1, record_id_eia) %>%
		write_parquet(FN$fn_id[i])
	
	TranchePredictors <- TrancheJoinedData %>%
		select(-record_id_ferc1, -record_id_eia)
	TranchePredictorsFactored <- collapse_factors(TranchePredictors)
	TranchePredictorsFactored %>%
		bake(recipe_fit, .) %>%
		write_parquet(FN$fn_x[i])
}
