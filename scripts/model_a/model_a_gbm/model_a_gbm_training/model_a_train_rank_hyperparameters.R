library(tidyverse)
library(skimr)
library(car)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

# Load data
fn_cv <- file.path(data_dir, '/working_data/model_a/model_a_training/gb_ray_tune/model_a_ann_hp_search_2_cv.csv')
fn_hp <- file.path(data_dir, '/working_data/model_a/model_a_training/gb_ray_tune/model_a_ann_hp_search.csv')

fn_model_a_gbm_hp <- file.path(data_dir, '/working_data/model_a/model_a_training/model_a_gbm_hp.csv')


CV <- read_csv(fn_cv, col_types = cols('hp_rank' = 'i', 'fold' = 'i'))
HP <- read_csv(fn_hp)

# Overall view
MeanLogLoss <-
	CV %>%
	group_by(hp_rank) %>%
	summarize(mean_log_loss = mean(log_loss)) %>%
	ungroup %>%
	mutate(
		rank = rank(mean_log_loss)
	) %>%
	arrange(rank)

MedianLogLoss <-
	CV %>%
	group_by(hp_rank) %>%
	summarize(median_log_loss = median(log_loss)) %>%
	ungroup %>%
	mutate(
		rank = rank(median_log_loss)
	) %>%
	arrange(rank)
MeanLogLoss %>% head
MedianLogLoss %>% head

# 6 and 3 are the candidates to beat
CV %>% 
	group_by(hp_rank) %>%
	summarize(
		mean_log_loss = mean(log_loss),
		median_log_loss = median(log_loss)
	) %>%
	ungroup %>%
	ggplot(aes(x = mean_log_loss, y = median_log_loss)) +
	geom_label(aes(label = hp_rank))

# I like model 6 from these diagrams
CV %>%
	select(-fold) %>%
	gather(variable, value, -hp_rank) %>%
	mutate(
		hp_rank = ordered(hp_rank),
		color = hp_rank %in% c(6, 8)
		) %>%
	ggplot(aes(x = hp_rank, y = value, fill = color)) +
	geom_boxplot() +
	facet_wrap(~variable, scales = 'free') +
	scale_fill_manual(values = c('white', 'dodgerblue')) +
	theme(legend.position = 'none')

# In fact, models 6 and 8 are nearly identical!
HP %>%
	filter(rank %in% c(6, 8)) %>%
	select(rank, starts_with('config')) %>%
	rename_all(str_replace, 'config/', '')

HP %>%
	filter(rank == 6) %>%
	select(rank, starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') %>%
	select(verbose, num_trees, learning_rate, min_data_in_leaf, objective, early_stopping_round, metrics) %>%
	write_csv(fn_model_a_gbm_hp)

# The hyperparameters in models 6 and 8 are also right in the middle
# of their distributions, which suggests that the optimizer was
# honing in thereabouts.
HP %>%
	select(starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') %>%
	select(num_trees, learning_rate, min_data_in_leaf) %>%
	car::scatterplotMatrix()
