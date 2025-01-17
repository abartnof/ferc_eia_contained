library(tidyverse)
library(skimr)

fn_metrics <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/ann/metrics_cross_validation_of_best_candidates_ann.csv'
fn_history <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_a/train/ann/history_cross_validation_of_best_candidates_ann.csv'

Metrics <- read_csv(fn_metrics)
History = read_csv(fn_history, col_types = cols('hp_rank' = 'i', 'fold' = 'i', 'epoch' = 'i'))

# Boxplots
Metrics %>%
	select(-fold) %>%
	gather(variable, value, -hp_rank) %>%
	mutate(hp_rank = ordered(hp_rank)) %>%
	ggplot(aes(x = hp_rank, y = value)) +
	geom_boxplot() +
	facet_wrap(~variable, scales='free_y') +
	labs(x = 'Model', y = 'Value')

# Ranked values
RankedMetrics <-
	Metrics %>%
	select(-fold) %>%
	mutate(log_loss = -log_loss) %>%  # only value we're trying to minimize
	gather(variable, value, -hp_rank) %>%
	group_by(variable, hp_rank) %>%
	summarize(avg = mean(value)) %>%
	ungroup %>%
	group_by(variable) %>%
	mutate(
		inverse_rank = rank(-avg),
		true_rank = rank(avg),
		color = inverse_rank == 1,
		hp_rank = factor(hp_rank)
		) %>%
	ungroup

RankedMetrics %>%
	ggplot(aes(x = hp_rank, y = true_rank)) +
	geom_col(aes(fill = color)) +
	facet_wrap(~variable) +
	geom_label(aes(label = inverse_rank), position = position_stack(vjust = .5)) +
	scale_fill_manual(values = c('darkgrey', 'dodgerblue')) +
	theme(legend.position = 'none',
				panel.grid.major.x = element_blank(), 
				axis.text.y = element_blank(), 
				axis.ticks = element_blank()) +
	labs(y = 'Rank of mean value', x = 'Model')
	
# History
MinBC <-
	History %>%
	select(hp_rank, fold, epoch, binary_crossentropy, val_binary_crossentropy) %>% 
	filter(hp_rank == 1L) %>%
	group_by(fold) %>%
	mutate(
		val_min_loss = case_when(
			(epoch >= 10) & (val_binary_crossentropy == min(val_binary_crossentropy)) ~ val_binary_crossentropy,
			TRUE ~ NA_real_),
		val_min_epoch = if_else(!is.na(val_min_loss), epoch, NA_integer_)
		) %>%
	ungroup

	MinBC %>%
	ggplot(aes(x = epoch)) +
	geom_line(aes(y = binary_crossentropy, group = fold)) +
	geom_line(aes(y = val_binary_crossentropy, group = fold), color = 'blue') +
	geom_point(aes(x = val_min_epoch, y = val_min_loss)) +
	facet_wrap(~fold)

# mean epoch is 21, median epoch is 22
MinBC %>%
	drop_na %>%
	select(val_min_epoch) %>%
	summarize(mean = mean(val_min_epoch), median = median(val_min_epoch))

# Modal epoch is 22
MinBC %>%
	drop_na %>%
	count(val_min_epoch) %>%
	arrange(desc(n))

# Use model at hp_rank 1, at epoch 22
