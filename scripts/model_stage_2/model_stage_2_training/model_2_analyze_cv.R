library(tidyverse)
library(skimr)
# library(car)
library(psych)
library(corrplot)

#### ELT ####
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_out <- file.path(data_dir, 'working_data/model_second_stage/model_second_stage_training/model_second_stage_gbm_hp.csv')
dir_cv <- file.path(data_dir, 'working_data/model_second_stage/model_second_stage_training/cv_results')
list_fn <- list.files(dir_cv, full.names=TRUE)
fn_hp <- file.path(data_dir, 'working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_03_21/gbm_grid_2025_03_21.csv')

fn_stats_out <- file.path(data_dir, '/output_data/stats/stats_stage_2.csv')
fn_splot_out <- file.path(data_dir, '/output_data/stats/splot_stage_2.png')
fn_boxplot_out <- file.path(data_dir, '/output_data/stats/boxplot_stage_2.png')


# Load raytune optuna hyperparameter search; create scatterplot matrix
HP <-
	read_csv(fn_hp) %>%
	select(rank, order, binary_logloss, auc, `config/num_trees`, `config/learning_rate`, `config/min_data_in_leaf`) %>%
	rename_all(str_replace, 'config/', '')
HP


png(filename=fn_splot_out);  HP %>%
	select(binary_logloss, auc, num_trees, learning_rate) %>%
	pairs.panels(., main='Hyperparameter search: Model stage 2'); dev.off()


# Check out CV results
TempCV <- data.frame()
for (fn in list_fn){
	TempCV <- TempCV %>% bind_rows(read.csv(fn))
}
TempCV <- as_tibble(TempCV)
ID <-
	TempCV %>%
	distinct(num_trees, min_data_in_leaf, learning_rate) %>%
	rowid_to_column(var = 'id')
CV <-
	TempCV %>%
	left_join(ID, by = c('num_trees', 'min_data_in_leaf', 'learning_rate')) %>%
	relocate('id')

# QC: All hyperparameter sets got 5 tests, which is correct
CV %>%
	count(id) %>% 
	distinct(n)

# ID #2 looks great to me
boxplots <-
	CV %>%
	select(id, precision, recall, log_loss, roc_auc) %>%
	mutate(
		id = ordered(id),
		color = if_else(id == 2L, 'Selected', 'Not selected')
	) %>%
	gather(variable, value, -id, -color) %>%
	ggplot(aes(x = id, y = value, fill = color)) +
	geom_boxplot() +
	facet_wrap(~variable, scales = 'free') +
	scale_fill_manual(values = c('white', 'dodgerblue')) +
	labs(x = 'Model', y = '', title = 'Cross-validation of model 2', fill = '') +
	theme(
		legend.position = 'bottom',
		axis.ticks.x = element_blank()
	)
plot(boxplots)
ggsave(filename=fn_boxplot_out, plot=boxplots)
#

#### Export chosen hyperparameters
# Use model id 2:

# num_trees: 555
# min_data_in_leaf: 147
# learning_rate: 0.0144
CV %>%
	filter(id == 2) %>%
	distinct(num_trees, learning_rate, min_data_in_leaf, early_stopping_round)

CV %>%
	filter(id == 2L) %>%
	select(fold_num, precision, recall, log_loss, roc_auc) %>%
	write_csv(fn_stats_out)

CV %>%
	inner_join(Contenders, by = 'id') %>%
	filter(distance_rank == 1) %>%
	distinct(num_trees, min_data_in_leaf, learning_rate, early_stopping_round) %>%
	write_csv(fn_out)

