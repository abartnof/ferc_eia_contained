# Load the ground-truth mappings from FERC:EIA; 
# ensure that every row in it is something that we can use in this model
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof

library(tidyverse)
library(skimr)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
dir_input <- file.path(data_dir, '/input_data/')
dir_working <- file.path(data_dir, '/working_data/')

fn_raw_training_data <- file.path(dir_input, 'eia_ferc1_training.csv')
fn_eia_plant_parts <- file.path(dir_input, 'eia_plant_parts.RDS')
fn_ferc_steam <- file.path(dir_input, 'ferc_steam.RDS')
fn_clean_training_data <- file.path(dir_working, 'positive_matches.RDS')

EiaPlantParts <- readRDS(fn_eia_plant_parts)
FercSteam <- readRDS(fn_ferc_steam)

RawTrainingData <- read_csv(fn_raw_training_data, col_types = 'ccccc') %>%
	select(record_id_ferc1, record_id_eia)

#### Filter to cases where the training data includes 'legal' data based on our rules in the prev step ####
# print('Num. cases where the training data can be found in our predefined hypothesis space:')
# RawTrainingData %>%
# 	mutate(
# 		is_ferc1_legal = record_id_ferc1 %in% FercSteam$record_id_ferc1,
# 		is_eia_legal = record_id_eia %in% EiaPlantParts$record_id_eia
# 	) %>%
# 	count(is_ferc1_legal, is_eia_legal) %>%
# 	mutate(prop = prop.table(n))

LegalTrainingData <-
	RawTrainingData %>%
	mutate(
		is_ferc1_legal = record_id_ferc1 %in% FercSteam$record_id_ferc1,
		is_eia_legal = record_id_eia %in% EiaPlantParts$record_id_eia
	) %>%
	filter(is_ferc1_legal & is_eia_legal) %>%
	select(record_id_ferc1, record_id_eia)

#### Note the report year for each row in the legal training data ####
ContextEia <-
	EiaPlantParts %>% 
	select(record_id_eia, report_year) %>%
	rename(report_year_eia = report_year)

ContextFerc <-
	FercSteam %>% 
	select(record_id_ferc1, report_year) %>%
	rename(report_year_ferc1 = report_year)

GetBothReportYears <-
	LegalTrainingData %>%
	left_join(ContextEia, by = 'record_id_eia') %>%
	left_join(ContextFerc, by = 'record_id_ferc1')

# 
print('Num. cases where both FERC and EIA have missing years:')
GetBothReportYears %>%
	mutate(is_missing = is.na(report_year_eia) & is.na(report_year_ferc1)) %>% 
	count(is_missing) %>%
	mutate(prop = prop.table(n))

print('Num. cases where either FERC or EIA has a missing year:')
GetBothReportYears %>%
	mutate(is_missing = is.na(report_year_eia)|is.na(report_year_ferc1)) %>% 
	count(is_missing) %>%
	mutate(prop = prop.table(n))

print('Num. cases where the two sources disagree on report year:')
GetBothReportYears %>%
	drop_na(report_year_eia, report_year_ferc1) %>%
	mutate(is_disagreement = report_year_eia != report_year_ferc1) %>%
	count(is_disagreement) %>%
	mutate(prop = prop.table(n))

# Since they never disagree, use coalesce() to snag a report year for each pair.
# Drop any potential rows without report years.
TrainingDataWithReportYear <-
	GetBothReportYears %>%
	mutate(report_year = coalesce(report_year_ferc1, report_year_eia)) %>%
	select(record_id_ferc1, record_id_eia, report_year) %>%
	drop_na(report_year)

#### Final exclusions ####
# Omit any rows that we've prima facie excluded from the EIA yearly plant parts 
# table	

# TrainingDataWithReportYear %>%
# 	mutate(is_eia_record_ok = record_id_eia %in% EiaPlantParts$record_id_eia) %>%
# 	count(is_eia_record_ok)

# There is one case in which two FERC records match to a single EIA record-- 
# omit this
print('Num. cases where multiple FERC records match to a single EIA record, which is illegal:')
TrainingDataWithReportYear %>%
	distinct(record_id_ferc1, record_id_eia) %>%
	count(record_id_eia, name = 'num_matches_for_this_eia_record') %>%
	count(num_matches_for_this_eia_record)

TrainingDataWithReportYear %>%
	distinct(record_id_ferc1, record_id_eia, .keep_all=TRUE) %>%
	add_count(record_id_eia) %>%
	filter(n == 1L) %>%
	select(-n) %>%
	saveRDS(fn_clean_training_data)
