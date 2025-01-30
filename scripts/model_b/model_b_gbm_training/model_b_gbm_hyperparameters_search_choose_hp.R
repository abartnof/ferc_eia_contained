library(tidyverse)

fn_cv <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_b/model_b_training/gb_ray_tune/model_b_ann_hp_search_2_cv.csv'
CV <- read_csv(fn_cv, col_types = cols('hp_rank' = 'i', 'fold' = 'i'))

fn_hp <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_b/model_b_training/gb_ray_tune/model_b_ann_hp_search.csv' 
HP <- read_csv(fn_hp)

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
	gather(variable, value, -hp_rank, -fold) %>%
	mutate(
		hp_rank = factor(hp_rank),
		color = hp_rank == 3L
		) %>%
	ggplot(aes(x = hp_rank, y = value, fill = color)) +
	geom_boxplot() +
	facet_wrap(~variable, scales = 'free')

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
		color = hp_rank == 3L,
		hp_rank = ordered(hp_rank)
	) %>%
	ggplot(aes(x = hp_rank, color = color)) +
	geom_point(aes(y = mid)) +
	geom_errorbar(aes(ymin = low, ymax = high), width = 0) +
	facet_wrap(~variable, scales = 'free') +
	scale_color_manual(values = c('darkgrey', 'dodgerblue'))

# Go with rank == 3. rank == 2 is very similar, but with more
# variance in its metrics.
# HP %>%
# 	filter(rank == 2) %>%
# 	select(starts_with('config')) %>%
# 	rename_all(str_replace, 'config/', '') 

HP %>%
	filter(rank == 3) %>%
	select(starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') 

# num_trees:266
# learning_rate:0.0105
# min_data_in_leaf:42
# verbose:-1
# objective:'binary'
# early_stopping_round:-1
# metrics:'binary_logloss', 'auc'