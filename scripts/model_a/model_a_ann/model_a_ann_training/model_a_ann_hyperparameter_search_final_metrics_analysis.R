library(tidyverse)
library(skimr)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_model_a_training_ann <- file.path(data_dir, '/working_data/model_a/model_a_training/ann_ray_tune/')

fn_metrics <- file.path(
	dir_model_a_training_ann, 
	'metrics_cross_validation_of_best_candidates_ann.csv'
)

fn_history <- file.path(
	dir_model_a_training_ann, 
	'history_cross_validation_of_best_candidates_ann.csv'
)

fn_hp <- file.path(
	dir_model_a_training_ann, 
	'model_a_ann_hp_search.csv'
)


Metrics <- read_csv(fn_metrics, col_types = cols('hp_rank' = 'i', 'fold' = 'i'))
History <- read_csv(fn_history, col_types = cols('hp_rank' = 'i', 'fold' = 'i', 'epoch' = 'i'))
HP <- read_csv(fn_hp)

# Boxplots
Metrics %>%
	select(-fold) %>%
	gather(variable, value, -hp_rank) %>%
	mutate(hp_rank = ordered(hp_rank)) %>%
	ggplot(aes(x = hp_rank, y = value)) +
	geom_boxplot() +
	facet_wrap(~variable, scales='free_y') +
	labs(x = 'Model', y = 'Value')

# hp_rank == 3 is our best model
Metrics %>%
	group_by(hp_rank) %>%
	summarize(
		mean_log_loss = mean(log_loss),
	) %>%
	ungroup %>%
	arrange(mean_log_loss) %>%
	head

Metrics %>%
	group_by(hp_rank) %>%
	summarize(
		median_log_loss = median(log_loss),
	) %>%
	ungroup %>%
	arrange(median_log_loss) %>%
	head

# History
MungedTraining <-
	History %>%
	filter(hp_rank == 3L) %>%
	select(fold, epoch, binary_crossentropy, val_binary_crossentropy) %>%
	mutate(
		val_binary_crossentropy_adj = case_when(
			epoch >= 10L ~ val_binary_crossentropy,
			TRUE ~ NA_real_
		)
	) %>%
	group_by(fold) %>%
	mutate(
		target_epoch = case_when(
			val_binary_crossentropy_adj == min(val_binary_crossentropy_adj, na.rm=TRUE) ~ epoch,
			TRUE ~ NA_integer_
		)
	) %>%
	ungroup

PlotMeLines <-
	MungedTraining %>%
	select(fold, epoch, binary_crossentropy, val_binary_crossentropy) %>%
	rename('Training' = binary_crossentropy, 'Validation' = val_binary_crossentropy) %>%
	gather(variable, value, -fold, -epoch)

PlotMePoints <-
	MungedTraining %>%
	select(fold, target_epoch, val_binary_crossentropy) %>%
	drop_na

PlotMeLines %>%
	ggplot() +
	geom_vline(xintercept = 10, linetype = 'dotted') +
	geom_line(aes(x = epoch, y = value, group = variable, color = variable)) +
	geom_point(data = PlotMePoints, aes(x = target_epoch, y = val_binary_crossentropy)) +
	facet_wrap(~fold) +
	scale_color_manual(values = c('grey30', 'dodgerblue')) +
	labs(
		y = 'Binary crossentropy', 
		x = 'Epoch', 
		color = 'Data', 
		title = 'Target number of epochs for chosen model'
	)

#### Conclusion ####
# model n.3

# There is no 'modal' number of epochs we should train for. 
# Choose 20, since 18.6 epochs is impossible.
PlotMePoints %>%
	count(target_epoch)
median(PlotMePoints$target_epoch)
mean(PlotMePoints$target_epoch)


# dropout_1: 0.000120
# dropout_2: 0.0633 
# relu_1: 33
# relu_2: 20

HP %>% 
	filter(rank == 3L) %>%
	select(starts_with('config')) %>%
	rename_all(str_replace, 'config/', '')

