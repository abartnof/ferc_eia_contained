library(tidyverse)
library(skimr)
library(arrow)
library(caret)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_y_fit_model_a <- file.path(data_dir, 'working_data/model_z/y_fit_model_a.parquet')
fn_y_fit_model_b <- file.path(data_dir, 'working_data/model_z/y_fit_model_b.parquet')
fn_combos <- file.path(data_dir, '/working_data/matches_and_mismatches.parquet')
fn_ferc_to_fold <- file.path(data_dir, '/working_data/ferc_to_fold.parquet')

YFitModelA <- read_parquet(fn_y_fit_model_a)
YFitModelB <- read_parquet(fn_y_fit_model_b)
Combos <- read_parquet(fn_combos) %>%
	select(record_id_ferc1, record_id_eia, is_match)
# FercToFold <- read_parquet(fn_ferc_to_fold)

GroundTruth <-
	YFit %>%
	select(starts_with('record_'), fold) %>%
	left_join(Combos, by = join_by(record_id_ferc1, record_id_eia))

YFit <-
	YFitModelA %>%
	select(record_id_ferc1, record_id_eia, y_fit_a_ann, y_fit_a_gbm) %>%
	left_join(YFitModelB, by = join_by(record_id_ferc1, record_id_eia)) %>%
	relocate(record_id_ferc1, record_id_eia, fold)

YFitLong <-
	YFit %>%
	pivot_longer(cols = starts_with('y_fit'), names_to = 'model', values_to = 'y_fit', names_prefix = 'y_fit_')

get_gof <- function(InputGroundTruth, InputMappings){
	InputGroundTruth %>%
	left_join(InputMappings, join_by(record_id_ferc1, record_id_eia, fold), relationship = 'one-to-one') %>%
	mutate(
		is_fit = replace_na(is_fit, FALSE),
		is_match = ordered(is_match, levels = c(TRUE, FALSE)),
		is_fit = ordered(is_fit, levels = c(TRUE, FALSE))
	) %>%
	select(fold, is_match, is_fit) %>%
	group_by(fold) %>%
	nest %>%
	mutate(
		is_match = map(data, 'is_match'),
		is_fit = map(data, 'is_fit'),
	) %>%
	mutate(
		cm = map2(is_fit, is_match, ~confusionMatrix(data = .x, reference = .y, mode = 'everything', positive = 'TRUE')),
		cm = map(cm, 'byClass'),
		cm = map(cm, enframe, name = 'metric')
	) %>%
	select(fold, cm) %>%
	unnest(cm) %>%
	ungroup %>%
	filter(metric %in% c('Sensitivity', 'Specificity', 'Precision', 'Recall'))
}


# Null model 1: Modal responses
	# Find each model's 'top pick' (ie the eia record with the highest y_fit)	
		# For each ferc record, take any record that was chosen twice.
		# If no record was chosen twice, 
			# pick the one that was chosen once, 
			# albeit with the highest y_fit

CteModalCandidates <-
	YFitLong %>%
	group_by(model, fold, record_id_ferc1) %>%
	slice_max(y_fit, na_rm = TRUE, with_ties = TRUE) %>%  # all 'top' suggestions, per model x fold
	ungroup %>%
	group_by(fold, record_id_ferc1, record_id_eia) %>%
	summarize(
		num_votes = n(),
		max_y_fit = max(y_fit)
	) %>%
	ungroup

Mappings1 <-
	CteModalCandidates %>%
	group_by(fold, record_id_ferc1) %>%
	slice_max(order_by = tibble(num_votes, max_y_fit), n = 1, with_ties = FALSE) %>%
	ungroup %>%
	select(fold, record_id_ferc1, record_id_eia) %>%
	mutate(is_fit = TRUE)
# Diagnostics- GOF
length(unique(GroundTruth$record_id_ferc1)) == nrow(Mappings1)
GOF1 <- get_gof(GroundTruth, Mappings1)
GOF1

# Null Model 2: Weighted Votes
# For each possible mapping, add all the models' y_fits.
# The EIA entry with the highest y_fit wins.

Mappings2 <-
	YFitLong %>%
	group_by(fold, record_id_ferc1, record_id_eia) %>%
	summarize(sum_y_fit = sum(y_fit)) %>%
	ungroup %>%
	group_by(fold, record_id_ferc1) %>%
	slice_max(sum_y_fit, n = 1, with_ties = FALSE) %>%
	ungroup %>%
	select(fold, record_id_ferc1, record_id_eia) %>%
	mutate(is_fit = TRUE)

length(unique(GroundTruth$record_id_ferc1)) == nrow(Mappings2)
GOF2 <- get_gof(GroundTruth, Mappings2)

# Null Model 3: Scaled weighted votes
# Same as above, but first, scale each fold x model's y_fits;
# by ipsotizing each model's y_fits, we no longer prefer models
# with generally higher y_fits
Mappings3 <-
	YFitLong %>%
	drop_na(y_fit) %>%
	group_by(fold, model) %>%
	mutate(y_fit_z = as.vector(scale(y_fit))) %>%
	ungroup %>%
	group_by(fold, record_id_ferc1, record_id_eia) %>%
	summarize(sum_y_fit_z = sum(y_fit_z)) %>%
	ungroup %>%
	group_by(fold, record_id_ferc1) %>%
	slice_max(sum_y_fit_z, n = 1, with_ties = FALSE) %>%
	ungroup %>%
	select(fold, record_id_ferc1, record_id_eia) %>%
	mutate(is_fit = TRUE)

length(unique(GroundTruth$record_id_ferc1)) == nrow(Mappings3)
GOF3 <- get_gof(GroundTruth, Mappings3)
#
bind_rows(
	GOF1 %>%
	mutate(model = 'Modal'),
	GOF2 %>%
	mutate(model = 'Weighted Avg'),
	GOF3 %>%
	mutate(model = 'Scaled Weighted Avg')
) %>%
mutate(model = ordered(model, c('Modal', 'Weighted Avg', 'Scaled Weighted Avg'))) %>%
ggplot(aes(x = model, y = value)) +
geom_boxplot() +
facet_wrap(~metric, scales = 'free_y')
	
#
# Bonus: each model x fold's suggestion
	
