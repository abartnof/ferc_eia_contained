# Choose which hyperparameters from the cross-validation 
# to use for the final model
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof

library(tidyverse)
library(skimr)
library(psych)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_model_a_training_ann <- file.path(data_dir, '/working_data/model_a/model_a_training/ann_ray_tune/')

fn_stats_out <- file.path(data_dir, '/output_data/stats/stats_a_ann.csv')
fn_splot_out <- file.path(data_dir, '/output_data/stats/splot_a_ann.png')
fn_boxplot_out <- file.path(data_dir, '/output_data/stats/boxplot_a_ann.png')

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

fn_model_a_ann_hp <- file.path(
	data_dir, 
	'/working_data/model_a/model_a_training/model_a_ann_hp.csv'
)


Metrics <- read_csv(fn_metrics, col_types = cols('hp_rank' = 'i', 'fold' = 'i'))
History <- read_csv(fn_history, col_types = cols('hp_rank' = 'i', 'fold' = 'i', 'epoch' = 'i'))
HP <- read_csv(fn_hp)

png(filename=fn_splot_out);  HP %>%
	select(binary_crossentropy, auc, starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') %>%
	pairs.panels(., main='Hyperparameter search: ANN A');  dev.off()

# Loss
# Pretty clear that hp_rank == 3 has the lowest log-loss
Metrics %>%
	group_by(hp_rank) %>%
	summarize(
		`Mean log-loss` = mean(log_loss),
		`Median log-loss` = median(log_loss),
	) %>%
	ungroup %>%
	ggplot(aes(x = `Mean log-loss`, y = `Median log-loss`, label = hp_rank)) +
	geom_label()

# Boxplots
boxplot <-
	Metrics %>%
	select(-fold) %>%
	gather(variable, value, -hp_rank) %>%
	mutate(
		hp_rank = ordered(hp_rank),
		color = if_else(hp_rank == 3, 'Selected', 'Not selected')
	) %>%
	ggplot(aes(x = hp_rank, y = value, fill = color)) +
	geom_boxplot() +
	facet_wrap(~variable, scales='free_y') +
	scale_fill_manual(values = c('white', 'dodgerblue')) +
	labs(x = 'Model', y = '', title = 'Cross-validation of ANN A', fill = '') +
	theme(
		legend.position = 'bottom',
		axis.ticks.x = element_blank())

plot(boxplot)
ggsave(plot=boxplot, filename=fn_boxplot_out)

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

# Find the optimal number of epochs in the history
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
	rename_all(str_replace, 'config/', '') %>%
	mutate(epochs = 20) %>%
	write_csv(fn_model_a_ann_hp)

# Model 3 is right in the middle of the optimizer's targets
HP %>%
	select(starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') %>%
	scatterplotMatrix()

Metrics %>%
	filter(hp_rank == 3) %>%
	select(fold, accuracy, roc_auc, log_loss, precision, recall) %>%
	write_csv(fn_stats_out)
