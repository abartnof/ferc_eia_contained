library(tidyverse)
library(arrow)
library(skimr)

#### ELT ####
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_y_fit_1_a_ann <- file.path(data_dir, '/working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_a_ann.parquet')
fn_y_fit_1_a_gbm <- file.path(data_dir, '/working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_a_gbm.parquet')
fn_y_fit_1_b_ann <- file.path(data_dir, '/working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_b_ann.parquet')
fn_y_fit_1_b_gbm <- file.path(data_dir, '/working_data/model_second_stage/model_second_stage_training/temp/y_fit_1_b_gbm.parquet')
fn_y_fit_2_gbm <-   file.path(data_dir, '/working_data/model_second_stage/model_second_stage_training/temp/y_fit_2.parquet')
fn_feature_importance <-   file.path(data_dir, '/working_data/model_second_stage/model_second_stage_training/temp/mod2_feature_importance.parquet')

fn_id <- file.path(data_dir, '/working_data/model_a/model_a_training/id.parquet')
fn_matches <- file.path(data_dir, 'working_data/positive_matches.RDS')

fn_ferc_steam <- file.path(data_dir, 'input_data/ferc_steam.RDS')
fn_ferc_utilities <- file.path(data_dir, 'input_data/ferc_utilities.RDS')
fn_eia <- file.path(data_dir, 'input_data/eia_plant_parts.RDS')


YFit1AAnn <- read_parquet(fn_y_fit_1_a_ann)
YFit1AGbm <- read_parquet(fn_y_fit_1_a_gbm)
YFit1BAnn <- read_parquet(fn_y_fit_1_b_ann)
YFit1BGbm <- read_parquet(fn_y_fit_1_b_gbm)
YFit2Gbm <- read_parquet(fn_y_fit_2_gbm)
FeatureImportance <- read_parquet(fn_feature_importance)
ID <- read_parquet(fn_id)
FercSteam <- readRDS(fn_ferc_steam)
FercUtilities <- readRDS(fn_ferc_utilities)

Eia <- readRDS(fn_eia)

Matches <- readRDS(fn_matches) %>%
	select(record_id_ferc1, record_id_eia) %>%
	mutate(is_match = TRUE)

# Moderate data cleaning, joining

CteEia <- Eia %>%
	select(record_id_eia, plant_part, plant_name_eia, utility_name_eia, net_generation_mwh) %>%
	rename(plant_part_eia = plant_part, net_generation_mwh_eia = net_generation_mwh) %>%
	relocate(record_id_eia, plant_name_eia, plant_part_eia, utility_name_eia)

CteFerc <-
	FercSteam %>%
	select(record_id_ferc1, plant_name_ferc1, utility_id_ferc1, net_generation_mwh) %>%
	left_join(FercUtilities, by = 'utility_id_ferc1') %>%
	select(record_id_ferc1, plant_name_ferc1, utility_name_ferc1, net_generation_mwh) %>%
	rename(net_generation_mwh_ferc1 = net_generation_mwh)

YFit <- bind_cols(YFit1AAnn, YFit1AGbm, YFit1BAnn, YFit1BGbm, YFit2Gbm)

# Find mappings
ModelsMappings <-
	ID %>%
	select(-report_year, -fold) %>%
	bind_cols(YFit) %>%
	gather(model, y_fit, -record_id_ferc1, -record_id_eia) %>%
	group_by(record_id_ferc1, model) %>%
	slice_max(y_fit, n = 1, with_ties = TRUE) %>%
	ungroup

# Joined Data
JoinedData <-
	ModelsMappings %>%
	left_join(Matches, by = c('record_id_ferc1', 'record_id_eia')) %>%
	mutate(is_match = replace_na(is_match, FALSE)) %>%
	left_join(CteFerc, by = 'record_id_ferc1') %>%
	left_join(CteEia, by = 'record_id_eia') %>%
	mutate(model = ordered(model),
				 net_gen_ratio = net_generation_mwh_ferc1 / net_generation_mwh_eia
				 net_gen_ratio = round(net_gen_ratio, 2)
  ) %>%
	relocate(record_id_ferc1, record_id_eia, model, y_fit, is_match, plant_name_ferc1, utility_name_ferc1, plant_name_eia, utility_name_eia, plant_part_eia, net_gen_ratio, net_generation_mwh_ferc1, net_generation_mwh_eia)

CteBadMatches <-
	JoinedData %>%
	filter(model == 'y_fit_2', !is_match) %>%
	distinct(record_id_ferc1)

BadMatches <-
	JoinedData %>%
	semi_join(CteBadMatches)

JoinedData %>%
	write_csv('~/Downloads/qc_joined_data_mappings.csv')

#### Feature Importance ####
FeatureImportance %>%
	ggplot(aes(x = colnames, y = feature_importance)) +
	geom_col() +
	coord_flip()

FeatureImportance %>%
	mutate(group = case_when(
		str_detect(colnames, 'rank') ~ 'Rank',
		str_detect(colnames, 'y_fit') ~ 'Y-Fit',
		TRUE ~ 'Other'
	)) %>%
	group_by(group) %>%
	summarize(importance = sum(feature_importance)) %>%
	ungroup %>%
	mutate(prop = importance/sum(importance)) %>%
	ggplot(aes(x = group, y = prop)) +
	geom_col()