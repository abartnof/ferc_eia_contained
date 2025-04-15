# Perform all of the metrics comparisons that will be 
# noted in the feature engineering
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof


# PART 1, FERC VALUES PER SE:

# ProductRecordIdFerc1ToIs100Percent$is_100_percent
# 	TRUE: 100% is found in the plant_name_ferc1
# 	FALSE: a percentage less than 100% is found in the plant_name_ferc1
# 	NA: no percentage is found in the plant_name_ferc1

# ProductPlantNameFerc1ToToken
#		19 tokens are looked for in the plant_name_ferc1, each one is represented in a column

# ProductRecordIdFerc1ToNum
# Nums: Note all of the numbers that are explicitly noted ('1' yields 1) or referenced 
	# ('1-3' yields 1,2,3).
# $does_plant_name_ferc1_contain_multiple_numbers
# 	TRUE: plant_name_ferc1 contains multiple numbers, eg (1-3), (1,2)
#		FALSE: plant_name_ferc1 contains a single number 
#		NA: no numbers are found in plant_name_ferc1
	

# PART 2, COMPARISONS BETWEEN FERC AND EIA VALUES:

# PlantNameLVDist
#		Join by plant_name_ferc1 and plant_name_eia, get:
#		plant_name_dist_lv: levenshtein distance
#		ratio_lv_to_plant_name_ferc1_len: normalize levenshtein distance to length of plant_name_ferc1

# UtilityNameLVDist
#		Join by utility_name_ferc1 and utility_name_eia, get:
#		utility_name_dist_lv: levenshtein distance
#		ratio_lv_to_utility_name_ferc1_len: normalize levenshtein distance to length of utility_name_ferc1

# TODO: capacity_margin = abs(1 - (capacity_mw_eia / capacity_mw_ferc1))
#		This will be done in the next step

library(tidyverse)
library(skimr)
library(arrow)
library(stringdist)

############################################################
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
############################################################

#### Read data ####
fn_combo <- file.path(data_dir, 'working_data/matches_and_mismatches.parquet')
fn_eia_plant_parts <-file.path(data_dir, 'input_data/eia_plant_parts.RDS')
fn_ferc_steam <-file.path(data_dir, 'input_data/ferc_steam.RDS')
fn_ferc_utility <-file.path(data_dir, 'input_data/ferc_utilities.RDS')

dir_out <- file.path(data_dir, 'working_data/model_b')
fn_ferc1_is_100 <- file.path(dir_out, 'product_record_id_ferc1_to_is_100_percent.parquet')
fn_ferc1_to_numbers <- file.path(dir_out, 'product_record_id_ferc1_to_numbers.RDS')
fn_ferc1_to_token <- file.path(dir_out, 'product_plant_name_ferc1_to_token.parquet')
fn_plant_name_lv_distance <- file.path(dir_out, 'product_plant_name_lv_distance.parquet')
fn_utility_name_lv_distance <- file.path(dir_out, 'product_utility_name_lv_distance.parquet')

Combo <- read_parquet(fn_combo)
EiaPlantParts <- readRDS(fn_eia_plant_parts)
FercSteam <- readRDS(fn_ferc_steam)
FercUtility <- readRDS(fn_ferc_utility)

FercContext <-
	FercSteam %>%
	select(record_id_ferc1, plant_name_ferc1, capacity_mw) %>%
	rename(capacity_mw_ferc1 = capacity_mw)
EiaContext <-
	EiaPlantParts %>%
	mutate(
		generator_integer = str_c(parse_integer(str_extract(generator_id, '\\d+'))),
	) %>%
	select(record_id_eia, plant_name_eia, generator_id, generator_integer, plant_part, capacity_mw) %>%
	rename(capacity_mw_eia = capacity_mw)

#### is_100_percent ####
	# Check if there's a % in the plant_name_ferc1.
	# NA means no number + percentage; true means 100%; false means any other number + percentage

ProductRecordIdFerc1ToIs100Percent <-
	FercSteam %>%
	select(record_id_ferc1, plant_name_ferc1) %>%
	mutate(
		substring_percent = str_extract(plant_name_ferc1, '\\d+\\.*\\d*%'),  # https://stackoverflow.com/questions/19252663/extracting-decimal-numbers-from-a-string
		substring_number = parse_number(str_replace(substring_percent, '%', '')),
		is_100_percent = substring_number == 100
	) %>%
	select(record_id_ferc1, is_100_percent)

ProductRecordIdFerc1ToIs100Percent %>%
	write_parquet(fn_ferc1_is_100)

#### Note singleton and sets of non-percentage numbers in string name ####
# RETURN:
# num, a nested list of integers within each ferc plant name

# ALSO RETURN:
# does_plant_name_ferc1_contain_multiple_numbers:
#	FALSE: multiple numbers
#	TRUE: 1 number
# NA: no numbers
# Motivation: if we know that the FERC record actually refers to multiple units, 
# then we can find the right EIA record scale!

# Number sequences
# Step 1: remove numbers with percentages, replace with X, 
	# so that we don't find spurious other ranges 
	# (eg 10 - 90% 1 should not be parsed as 10 - 1)

	# number(s)
	# OPTIONALLY followed by a decimal point and more numbers
	# OPTIONALLY followed by space(s)
	# followed by a percentage
pattern_percentage <- '\\d+[\\d\\.]*\\s*%'

# Step 2: remove #
# Step 3: Get ranges of numbers (should match "1 - 2 - 3", and extract 1, 3 as extrema)
	# At least one of: 
		# Integer(s)
		# OPTIONALLY followed by space(s)
		# followed by dash(es)
		# OPTIONALLY followed by space(s)
	# followed by integer(s)
	# OPTIONALLY followed by space(s)
	# NOT followed by X (placeholder for %)
pattern_seq <- '(\\d+\\s*-+\\s*)+\\d+\\s*(?!X)'

# Get all numbers in referenced sequences
FercToSeq <-
	FercSteam %>%
	distinct(plant_name_ferc1) %>%
	mutate(
		name_adj = str_replace_all(plant_name_ferc1, pattern_percentage, 'XX'),
		name_adj = str_replace_all(name_adj, '#', ''),
		seq_substring = str_extract_all(name_adj, pattern_seq),
		extracted_nums = map(seq_substring, str_extract_all, '\\d+'),
		) %>%
	unnest(extracted_nums) %>%
	unnest(extracted_nums) %>%
	mutate(extracted_nums = parse_integer(extracted_nums)) %>%
	group_by(plant_name_ferc1) %>%
	summarize(
		low = min(extracted_nums), 
		high = max(extracted_nums)
	) %>%
	ungroup %>%
	mutate(
		seq = map2(low, high, seq)
	) %>%
	unnest(seq) %>%
	select(plant_name_ferc1, seq) %>%
	rename(number = seq)

# Get singleton numbers
FercToSingletons <-
	FercSteam %>%
	distinct(plant_name_ferc1) %>%
	mutate(
		name_adj = str_replace_all(plant_name_ferc1, pattern_percentage, 'XX'),
		extracted_nums = map(name_adj, str_extract_all, '\\d+'),
	) %>%
	unnest(extracted_nums) %>%
	unnest(extracted_nums) %>%
	mutate(extracted_nums = parse_integer(extracted_nums)) %>%
	distinct(plant_name_ferc1, extracted_nums) %>%
	rename(number = extracted_nums)

# Join singletons and referenced sequences
FercToAllNumbers <-
	FercToSingletons %>%
	bind_rows(FercToSeq) %>%
	distinct(plant_name_ferc1, number) %>%
	arrange(plant_name_ferc1) %>%
	group_by(plant_name_ferc1) %>%
	nest %>%
	mutate(data = map(data, pull)) %>%
	ungroup

ProductRecordIdFerc1ToNum <-
	FercSteam %>%
	select(record_id_ferc1, plant_name_ferc1) %>%
	left_join(FercToAllNumbers) %>%
	mutate(
		amt_numbers = map_int(data, length),
		does_plant_name_ferc1_contain_multiple_numbers = case_when(
			amt_numbers == 0 ~ NA_integer_,
			amt_numbers == 1 ~ FALSE,
			amt_numbers > 1 ~ TRUE
		)
	) %>%
	select(record_id_ferc1, num = data, does_plant_name_ferc1_contain_multiple_numbers)
	
ProductRecordIdFerc1ToNum %>%
	write_rds(fn_ferc1_to_numbers)

#### String Metrics ####
# 1. Check for presence of key tokens from plant_name_ferc1
	# eg 'CC', 'Steam', etc-- other industry jargon that are instructive

punct_keep_pound <- "[^#[:^punct:]]"  # Remove all punctuation except for pound: https://stackoverflow.com/questions/8697079/remove-all-punctuation-except-apostrophes-in-r

ProductPlantNameFerc1ToToken <-
	FercSteam %>%
		distinct(plant_name_ferc1) %>%
		mutate(
			plant_name_ferc1_adj = str_replace_all(
				plant_name_ferc1, 
				punct_keep_pound, 
				' '
			),
			token = str_split(plant_name_ferc1_adj, '[\\s-_\\.]+')
		) %>%
		unnest(token) %>%
		mutate(
			token = str_replace_all(token, '\\d+', '')
		) %>%
		# filter(str_length(token) > 1L) %>%
		distinct(plant_name_ferc1, token) %>%
		mutate(
			token_adj = case_when(
				str_detect(token, 'com[a-z]*_*\\s*-*cyc') ~ 'cc',
				str_detect(token, 'gas[a-z]*_*\\s*-*tur') ~ 'gt',
				str_detect(token, 'com[a-z]*_*\\s*-*tur') ~ 'gt',
				token == 'cc' ~ token,
				token == 'com' ~ token,
				token == 'comb' ~ token,
				token == 'combined' ~ token,
				token == 'cyc' ~ 'cycle',
				token == 'cycle' ~ token,
				token == 'cycles' ~ 'cycle',
				token == 'diesel' ~ token,
				token == 'gas' ~ token,
				token == 'ct' ~ 'gt',
				token == 'gt' ~ token,
				token == 'nuclear' ~ token,
				token %in% 
					c('#', '#s', 'no', 'nos', 'num', 'nums', 
						'number', 'numbers') ~ 'number',
				token %in% c('peak', 'peaker', 'peaking') ~ 'peaker',
				token == 'share' ~ token,
				token == 'st' ~ token,
				token == 'station' ~ token,
				token == 'stations' ~ 'station',
				token == 'steam' ~ token,
				token == 'tot' ~ 'total',
				token == 'total' ~ token,
				token == 'turbine' ~ token,
				token == 'turbines' ~ 'turbine',
				token == 'unit' ~ token,
				token == 'units' ~ 'unit',
				token == 'wind' ~ token
			),
		) %>%
	drop_na(token_adj) %>%
	select(plant_name_ferc1, token_adj) %>%
	mutate(
		value = TRUE,
		token_adj = str_c('token__', token_adj)
	) %>%
	spread(token_adj, value, fill = FALSE)

ProductPlantNameFerc1ToToken %>%
	write_parquet(fn_ferc1_to_token)

# Calculate Levenshtein distance between plant names
JoinedPlantNames <-
	FercSteam %>%
	distinct(report_year, plant_name_ferc1) %>%
	inner_join(
		EiaPlantParts %>%
			distinct(report_year, plant_name_eia),
		by = 'report_year', 
		relationship = 'many-to-many'
	)

PlantNameLVDist <-
	JoinedPlantNames %>%
	distinct(plant_name_ferc1, plant_name_eia) %>%
	drop_na(plant_name_ferc1, plant_name_eia) %>%
	mutate(
		plant_name_dist_lv = stringdist(plant_name_ferc1, plant_name_eia, method = 'lv'),
		ratio_lv_to_plant_name_ferc1_len = (plant_name_dist_lv + 1)/ str_length(plant_name_ferc1) # add 1 so we don't get invalid values
	)
# PlantNameLVDist %>%
# 	select(where(is.numeric)) %>%
# 	skim

PlantNameLVDist %>%
	write_parquet(fn_plant_name_lv_distance)


# Calculate Levenshtein distance between utility names
JoinedUtilityNames <-
	FercSteam %>%
	left_join(FercUtility, by = 'utility_id_ferc1') %>%
	distinct(report_year, utility_name_ferc1) %>%
	inner_join(
		EiaPlantParts %>%
			distinct(report_year, utility_name_eia),
		by = 'report_year', 
		relationship = 'many-to-many'
	)

UtilityNameLVDist <-
	JoinedUtilityNames %>%
	distinct(utility_name_ferc1, utility_name_eia) %>%
	drop_na(utility_name_ferc1, utility_name_eia) %>%
	mutate(
		utility_name_dist_lv = stringdist(utility_name_ferc1, utility_name_eia, method = 'lv'),
		ratio_lv_to_utility_name_ferc1_len = (utility_name_dist_lv + 1)/ str_length(utility_name_ferc1) # add 1 so we don't get invalid values
	)
# UtilityNameLVDist %>%
# 	select(where(is.numeric)) %>%
# 	skim

UtilityNameLVDist %>%
	write_parquet(fn_utility_name_lv_distance)
