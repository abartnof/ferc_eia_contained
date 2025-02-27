library(tidyverse)
library(skimr)
library(car)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_model_b_training_ann <- file.path(data_dir, '/working_data/model_b/model_b_training/ann_ray_tune/')

fn_metrics <- file.path(
	dir_model_b_training_ann, 
	'metrics_cross_validation_of_best_candidates_ann.csv'
)

fn_history <- file.path(
	dir_model_b_training_ann, 
	'history_cross_validation_of_best_candidates_ann.csv'
)

fn_hp <- file.path(
	dir_model_b_training_ann, 
	'model_b_ann_hp_search.csv'
)

fn_model_b_ann_hp <- file.path(
	data_dir, 
	'/working_data/model_b/model_b_training/model_b_ann_hp.csv'
)



Metrics <- read_csv(fn_metrics, col_types = cols('hp_rank' = 'i', 'fold' = 'i'))
History <- read_csv(fn_history, col_types = cols('hp_rank' = 'i', 'fold' = 'i', 'epoch' = 'i'))
HP <- read_csv(fn_hp)

# Loss
# Model 7 looks promising, 13 in next place
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
Metrics %>%
	select(-fold) %>%
	gather(variable, value, -hp_rank) %>%
	mutate(
		color = case_when(
			hp_rank %in% c(7, 13) ~ hp_rank, 
			T ~ 99),
		hp_rank = ordered(hp_rank),
		color = ordered(color)
	) %>%
	ggplot(aes(x = hp_rank, y = value, fill = color)) +
	geom_boxplot() +
	facet_wrap(~variable, scales='free_y') +
	scale_fill_manual(values = c('dodgerblue', 'lightblue', 'white')) +
	labs(x = 'Model', y = 'Value') 

# hp_rank == 7 is our best model
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
	filter(hp_rank == 7L) %>%
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
	geom_vline(xintercept = 10, linetype = 'dotted') + # grace period for training
	geom_line(aes(x = epoch, y = value, group = variable, color = variable)) +
	geom_vline(xintercept = 14, linetype = 'dotted', color = 'red') +  # proposed target epoch
	geom_point(data = PlotMePoints, aes(x = target_epoch, y = val_binary_crossentropy)) +
	facet_wrap(~fold) +
	scale_color_manual(values = c('grey30', 'dodgerblue')) +
	labs(
		y = 'Binary crossentropy', 
		x = 'Epoch', 
		color = 'Data', 
		title = 'Target number of epochs for chosen model'
	)
PlotMePoints

#### Conclusion ####
# model n.7

# There is no 'modal' number of epochs we should train for. 
# Choose 14 epochs. 12 is the model response, but only barely.
# Account for the overall distribution with the median/mean ones
PlotMePoints %>%
	count(target_epoch)
median(PlotMePoints$target_epoch)
mean(PlotMePoints$target_epoch)


# dropout_1: 0.0177
# dropout_2: 0.00595
# relu_1: 56
# relu_2: 29

HP %>% 
	filter(rank == 7L) %>%
	select(starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') %>%
	mutate(epochs = 14L) %>%
	write_csv(fn_model_b_ann_hp)

HP %>%
	select(starts_with('config')) %>%
	rename_all(str_replace, 'config/', '') %>%
	scatterplotMatrix()
