library(tidyverse)
library(jsonlite)
library(skimr)

dir_hp <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/gb_ray_tune/' 
list_dirs <- list.dirs(dir_hp, recursive = FALSE, full.names = TRUE)

dir_out <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/gb_ray_tune'
fn_out <- file.path(dir_out, 'model_a_lightgbm_hyperparameters_result.csv')

AllJoinedResults <- tibble()
for (i in seq(1, length(list_dirs))){
	fn_result_json <- file.path(list_dirs[i], 'result.json')
	fn_params_json <- file.path(list_dirs[i], 'params.json')
	Result <- jsonlite::fromJSON(fn_result_json)
	Params <- jsonlite::fromJSON(fn_params_json)
	
	ResultFormatted <-
		tribble(
			~trial_id, ~binary_logloss, ~auc,
			Result$trial_id, Result$binary_logloss, Result$auc,
		)
	
	# ParamFormatted <-
	# 	Params %>%
	# 		enframe(name = 'param') %>%
	# 		filter(param %in% c('learning_rate', 'min_data_in_leaf', 'num_iterations')) %>%
	# 		unnest(value)
	ParamFormatted <-
		tribble(
			~learning_rate, ~min_data_in_leaf, ~num_iterations,
			Params$learning_rate, Params$min_data_in_leaf, Params$num_iterations,
		)
	
	JoinedResults <-
		ResultFormatted %>%
		bind_cols(ParamFormatted)
	AllJoinedResults <- AllJoinedResults %>% bind_rows(JoinedResults)
}

RankedResults <-
	AllJoinedResults %>%
	arrange(binary_logloss, desc(auc)) %>%
	rowid_to_column('rank') 

# Winning model:
RankedResults %>%
	head(1) %>%
	select(-trial_id)

RankedResults %>%
	write_csv(fn_out)