library(tidyverse)
library(skimr)
library(car)

#### ELT ####
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_out <- file.path(data_dir, 'working_data/model_second_stage/model_second_stage_training/model_second_stage_gbm_hp.csv')
dir_cv <- file.path(data_dir, 'working_data/model_second_stage/model_second_stage_training/cv_results')
list_fn <- list.files(dir_cv, full.names=TRUE)

TempCV <- data.frame()
for (fn in list_fn){
	TempCV <- TempCV %>% bind_rows(read.csv(fn))
}
TempCV <- as_tibble(TempCV)
ID <-
	CV %>%
	distinct(num_trees, min_data_in_leaf, learning_rate) %>%
	rowid_to_column(var = 'id')
CV <-
	TempCV %>%
	left_join(ID, by = c('num_trees', 'min_data_in_leaf', 'learning_rate')) %>%
	relocate('id')

#### QC ####
# All hyperparameter sets got 5 tests, which is correct
CV %>%
	count(id) %>% 
	distinct(n)

#### Analysis ####
CV %>%
	select(-id, -fold_num) %>%
	scatterplotMatrix()

# Divvy the data into a handful of points to focus on.
# Here, i'll take each hyperparameter set's avg precision and recall,
# plot this in 2d space, and find the distance from 
# perfect precision and recall (1,1).
Contenders <-
	CV %>%
	select(id, precision, recall) %>%
	gather(variable, value, -id) %>%
	group_by(id, variable) %>%
	summarize(
		mean = mean(value)
	) %>%
	spread(variable, mean) %>%
	rowwise() %>%
	mutate(
		dist = sqrt( 
			((precision-1)**2) + ((recall-1)**2)
			)
	) %>%
	ungroup %>%
	mutate(distance_rank = dense_rank(dist)) %>%
	filter(distance_rank <= 5) %>%
	rename(mean_precision = precision, mean_recall = recall) %>%
	select(id, distance_rank, mean_precision, mean_recall)

#### Diagrams ####

# Contenders only
CV %>%
	select(id, precision, recall, log_loss, roc_auc) %>%
	inner_join(Contenders, by = 'id') %>%
	select(id, distance_rank, precision, recall, log_loss, roc_auc) %>%
	gather(variable, value, -id, -distance_rank) %>%
	mutate_at(c('id', 'distance_rank'), ordered) %>%
	ggplot(aes(x = id, y = value, group = id, color = distance_rank)) +
	geom_boxplot() +
	facet_wrap(~variable, scales = 'free_y') +
	scale_color_brewer(palette = 'Spectral', direction = -1) +
	labs(x = 'Hyperparameter set ID', y = '', color = 'Rank', title = 'Best hyperparameters')

CV %>%
	select(id, precision, recall, log_loss, roc_auc) %>%
	left_join(Contenders, by = 'id') %>%
	select(id, distance_rank, precision, recall, log_loss, roc_auc) %>%
	gather(variable, value, -id, -distance_rank) %>%
	mutate_at(c('id', 'distance_rank'), ordered) %>%
	ggplot(aes(x = id, y = value, group = id, color = distance_rank)) +
	geom_boxplot() +
	facet_wrap(~variable, scales = 'free_y') +
	scale_color_brewer(palette = 'Spectral', direction = -1) +
	labs(x = 'Hyperparameter set ID', y = '', color = 'Rank', title = 'All hyperparameters')

CV %>%
	select(id, precision, recall, log_loss, roc_auc) %>%
	left_join(Contenders, by = 'id') %>%
	select(id, distance_rank, precision, recall, log_loss, roc_auc) %>%
	gather(variable, value, -id, -distance_rank) %>%
	group_by(id, distance_rank, variable) %>%
	summarize(mean_value = mean(value)) %>%
	ungroup %>%
	#mutate(distance_rank = replace_na(distance_rank, 99)) %>%
	mutate_at(c('id', 'distance_rank'), ordered) %>%
	ggplot(aes(x = id, y = mean_value, group = id, color = distance_rank)) +
	geom_point() +
	facet_wrap(~variable, scales = 'free_y') +
	scale_color_brewer(palette = 'Spectral', direction = -1, na.value = "grey50") +
	labs(x = 'Hyperparameter set ID', y = 'Mean', color = 'Rank', title = 'Mean values (all hyperparameters)')

# Model 2 looks good-- great recall, good precision
# QC-- look at the 2d space, see if these rankings made sense. 
# they did! model 2 looks very good here
CV %>%
	select(id, precision, recall) %>%
	gather(variable, value, -id) %>%
	group_by(id, variable) %>%
	summarize(
		mean = mean(value)
	) %>%
	spread(variable, mean) %>%
	rowwise() %>%
	mutate(
		dist = sqrt( 
			((precision-1)**2) + ((recall-1)**2)
		)
	) %>%
	ungroup %>%
	mutate(distance_rank = dense_rank(dist)) %>%
	rename(mean_precision = precision, mean_recall = recall) %>%
	select(id, distance_rank, mean_precision, mean_recall) %>%
	ggplot(aes(x = mean_precision, y = mean_recall, label = id)) +
	geom_text() 

CV %>%
	select(id, precision, recall) %>%
	gather(variable, value, -id) %>%
	group_by(id, variable) %>%
	summarize(
		mean = mean(value)
	) %>%
	spread(variable, mean) %>%
	rowwise() %>%
	mutate(
		dist = sqrt( 
			((precision-1)**2) + ((recall-1)**2)
		)
	) %>%
	ungroup %>%
	mutate(distance_rank = dense_rank(dist)) %>%
	rename(mean_precision = precision, mean_recall = recall) %>%
	select(id, distance_rank, mean_precision, mean_recall) %>%
	ggplot(aes(x = mean_precision, y = mean_recall, label = id)) +
	geom_text() +
	coord_cartesian(xlim = c(0.99, 1), ylim = c(0.994, 1))

#### Export chosen hyperparameters
CV %>%
	inner_join(Contenders, by = 'id') %>%
	filter(distance_rank == 1) %>%
	distinct(num_trees, min_data_in_leaf, learning_rate) %>%
	write_csv(fn_out)
