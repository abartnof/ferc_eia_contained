library(tidyverse)
library(skimr)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

# Load data
fn_cv <- file.path(data_dir, '/working_data/model_a/model_a_training/gb_ray_tune/model_a_ann_hp_search_2_cv.csv')
fn_hp <- file.path(data_dir, '/working_data/model_a/model_a_training/gb_ray_tune/model_a_ann_hp_search.csv')
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

# 5 and 8 look like the best candidates
CV %>% 
	group_by(hp_rank) %>%
	summarize(
		mean_log_loss = mean(log_loss),
		median_log_loss = median(log_loss)
	) %>%
	ungroup %>%
	ggplot(aes(x = mean_log_loss, y = median_log_loss)) +
	geom_label(aes(label = hp_rank))

#



CV %>%
	select(-fold) %>%
	gather(variable, value, -hp_rank) %>%
	mutate(
		hp_rank = ordered(hp_rank),
		color = hp_rank %in% c(5, 8)
		) %>%
	ggplot(aes(x = hp_rank, y = value, fill = color)) +
	geom_boxplot() +
	facet_wrap(~variable, scales = 'free') +
	scale_fill_manual(values = c('white', 'dodgerblue')) +
	theme(legend.position = 'none')

# I like model n.8, due to the low level of variance in its scores.
# Interestingly, models 5 and 8 are VERY similar.
HP %>%
	filter(rank %in% c(5, 8)) %>%
	select(rank, starts_with('config')) %>%
	rename_all(str_replace, 'config/', '')

HP %>%
	filter(rank == 8) %>%
	select(rank, starts_with('config')) %>%
	rename_all(str_replace, 'config/', '')

# Conclusion:
# num_trees = 482
# learning_rate = 0.0134
# min_data_in_leaf = 85