library(tidyverse)
library(skimr)
library(car)


data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_cv <- file.path(data_dir, 'working_data/model_b/model_b_training/gb_ray_tune/model_b_ann_hp_search_2_cv.csv')
CV <- read_csv(fn_cv, col_types = cols('hp_rank' = 'i', 'fold' = 'i'))

fn_hp <- file.path(data_dir, 'working_data/model_b/model_b_training/gb_ray_tune/model_b_ann_hp_search.csv')
HP <- read_csv(fn_hp)

fn_model_b_gbm_hp <- file.path(data_dir, '/working_data/model_b/model_b_training/fn_model_b_gbm_hp.csv')


# Look at 3, 2, 0, 1
CV %>%
	group_by(hp_rank) %>%
	summarize(
		median_log_loss = median(log_loss),
		mean_log_loss = mean(log_loss)
	) %>%
	ungroup %>%
	ggplot(aes(x = median_log_loss, y = mean_log_loss, label = hp_rank)) +
	geom_text()

CV %>%
	group_by(hp_rank) %>%
	summarize(
		median_log_loss = median(log_loss),
	) %>%
	ungroup %>%
	arrange(median_log_loss) %>%
	head


CV %>%
	gather(variable, value, -hp_rank, -fold) %>%
	mutate(
		hp_rank = factor(hp_rank),
		color = hp_rank %in% seq(0, 3)
		) %>%
	ggplot(aes(x = hp_rank, y = value, fill = color)) +
	geom_boxplot() +
	facet_wrap(~variable, scales = 'free')

# We'll go with model 3-- similar to 2, but with less variance
CV %>%
	gather(variable, value, -hp_rank, -fold) %>%
	group_by(hp_rank, variable) %>%
	summarize(
		low = quantile(value, 0.25),
		mid = median(value),
		high = quantile(value, 0.75)
	) %>%
	ungroup %>%
	mutate(
		color = hp_rank %in% seq(0, 3),
		hp_rank = ordered(hp_rank)
	) %>%
	ggplot(aes(x = hp_rank, color = color)) +
	geom_point(aes(y = mid)) +
	geom_errorbar(aes(ymin = low, ymax = high), width = 0) +
	facet_wrap(~variable, scales = 'free') +
	scale_color_manual(values = c('darkgrey', 'dodgerblue'))

# All of the top-performing models are very similar
HP %>%
	filter(rank %in% seq(0, 3)) %>%
	select(rank, starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') 

HP %>%
	select(starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') %>%
	select(num_trees, learning_rate, min_data_in_leaf) %>%
	car::scatterplotMatrix()

HP %>%
	filter(rank == 3L) %>%
	select(rank, starts_with('config')) %>%
	rename_all(str_replace, 'config/', '')  %>%
	select(verbose, num_trees, learning_rate, min_data_in_leaf, objective, early_stopping_round, metrics) %>%
	write_csv(fn_model_b_gbm_hp)
