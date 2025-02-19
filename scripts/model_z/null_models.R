library(tidyverse)
library(skimr)
library(arrow)
library(caret)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_y_fit_model_a <- file.path(data_dir, 'working_data/model_z/y_fit_model_a.parquet')
fn_y_fit_model_b <- file.path(data_dir, 'working_data/model_z/y_fit_model_b.parquet')
# fn_matches <- file.path(data_dir, '/working_data/positive_matches.RDS')
fn_combos <- file.path(data_dir, '/working_data/matches_and_mismatches.parquet')

YFitModelA <- read_parquet(fn_y_fit_model_a)
YFitModelB <- read_parquet(fn_y_fit_model_b)
Combos <- read_parquet(fn_combos) %>%
	select(record_id_ferc1, record_id_eia, is_match)

# Unfortunately, model A and model B define splits differently, 
# so they can't be combined


# Matches <- readRDS(fn_matches) %>%
# 	select(record_id_ferc1, record_id_eia)

# Null model 1: Modal responses
	# Find each model's 'top pick' (ie the eia record with the highest y_fit)	
		# For each ferc record, take any record that was chosen twice.
		# If no record was chosen twice, 
			# pick the one that was chosen once, 
			# albeit with the highest y_fit
CteModalCandidates <-
	YFitLong %>%
	drop_na(y_fit) %>%
	group_by(fold_num, model, record_id_ferc1) %>%
	slice_max(y_fit, na_rm = TRUE, with_ties = TRUE) %>%  # all 'top' suggestions, per model x fold
	ungroup %>%
	group_by(fold_num, record_id_ferc1, record_id_eia) %>%
	summarize(
		num_votes = n(),
		max_y_fit = max(y_fit)
	) %>%
	ungroup

Mappings1 <-
	CteModalCandidates %>%
	group_by(fold_num, record_id_ferc1) %>%
	slice_max(order_by = tibble(num_votes, max_y_fit), n = 1, with_ties = FALSE) %>%
	ungroup %>%
	select(fold_num, record_id_ferc1, record_id_eia) %>%
	mutate(is_fit = TRUE)

Combos
YFitLong %>%
	distinct(record_id_ferc1, record_id_eia, fold_num)
#


# Diagnostics1 <-
	CteHypothesisSpace %>%
	left_join(CteTrue, by = c('record_id_ferc1', 'record_id_eia')) %>%
	left_join(Mappings1, by = c('record_id_ferc1', 'record_id_eia', 'fold_num')) %>%
	mutate_at(c('is_true', 'is_fit'), replace_na, FALSE) %>%
	mutate(
		is_true = factor(is_true, levels = c(TRUE, FALSE)),
		is_fit = factor(is_fit, levels = c(TRUE, FALSE)),
	) %>%
	select(fold_num, is_true, is_fit) %>%
	group_by(fold_num) %>%
	nest %>%
	mutate(
		cm = map(data, ~with(., confusionMatrix(data = is_fit, reference = is_true, positive = 'TRUE'))),
		cm = map(cm, ~.$byClass),
		'precision' = map(cm, 'Precision'),
		'recall' = map(cm, 'Recall')
	) %>%
		unnest(c(precision, recall))
	# unnest(c(variable, value)) %>%
	# select(fold_num, variable, value) %>%
	# ungroup %>%
	# filter(variable %in% c('Precision', 'Recall', 'F1'))
	

CteHypothesisSpace %>%
	left_join(CteTrue, by = c('record_id_ferc1', 'record_id_eia')) %>%
	left_join(Mappings1, by = c('record_id_ferc1', 'record_id_eia', 'fold_num')) %>%
	mutate_at(c('is_true', 'is_fit'), replace_na, FALSE) %>%
	mutate(
		is_true = factor(is_true, levels = c(TRUE, FALSE)),
		is_fit = factor(is_fit, levels = c(TRUE, FALSE)),
	) %>%
	select(fold_num, is_true, is_fit) %>%
	filter(fold_num == 0) %>%
	with(., confusionMatrix(data = is_fit, reference = is_true, positive = 'TRUE')) %>%
	.$byClass
?confusionMatrix
#
	

	

# Null Model 2: Weighted Votes
# For each possible mapping, add all the models' y_fits.
# The EIA entry with the highest y_fit wins.
Mappings2 <-
	YFitLong %>%
	drop_na(y_fit) %>%
	group_by(record_id_ferc1, record_id_eia, fold_num) %>%
	summarize(sum_y_fit = sum(y_fit)) %>%
	ungroup %>%
	group_by(fold_num, record_id_ferc1) %>%
	slice_max(sum_y_fit, n = 1, with_ties = FALSE) %>%
	ungroup

# Null Model 3: Scaled weighted votes
# Same as above, but first, scale each fold x model's y_fits;
# by ipsotizing each model's y_fits, we no longer prefer models
# with generally higher y_fits
Mappings3 <-
	YFitLong %>%
	drop_na(y_fit) %>%
	group_by(fold_num, model) %>%
	mutate(y_fit_z = as.vector(scale(y_fit))) %>%
	ungroup %>%
	group_by(fold_num, record_id_ferc1, record_id_eia) %>%
	summarize(sum_y_fit_z = sum(y_fit_z)) %>%
	ungroup %>%
	group_by(fold_num, record_id_ferc1) %>%
	slice_max(sum_y_fit_z, n = 1, with_ties = FALSE) %>%
	ungroup
#

# Bonus: each model x fold's suggestion
YFitLong %>%
	
