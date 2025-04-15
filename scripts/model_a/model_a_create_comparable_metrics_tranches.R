# Compare easily comparable metrics
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof


# Use these locations to make the script work on your machine
dir_scripts <- '~/Documents/rmi/rematch_ferc_eia_pixi/ferc_eia/scripts/'
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

# Source script
fn_script <- file.path(dir_scripts, '/model_a/model_a_create_comparable_metrics_functions.R')
source(fn_script)

# Point to other files
dir_input <- file.path(data_dir, '/input_data/')
dir_working <- file.path(data_dir, '/working_data/')
# dir_working_model_a_training <- file.path(
# 	data_dir,
# 	'/working_data/model_a/model_a_training/'
# )

dir_tranches <- file.path(dir_working, '/tranches_ferc_to_eia')
dir_output <- file.path(dir_working, '/model_a/model_a_comparable_metrics')

fn_eia_plant_parts <- file.path(dir_input, 'eia_plant_parts.RDS')
fn_ferc_steam <- file.path(dir_input, 'ferc_steam.RDS')
fn_ferc_utilities <- file.path(dir_input, 'ferc_utilities.RDS')

# Load files
FercSteam <- readRDS(fn_ferc_steam) %>% lazy_dt
FercUtilities <- readRDS(fn_ferc_utilities) %>% lazy_dt
EiaPlantParts <- readRDS(fn_eia_plant_parts) %>% lazy_dt

#### Run Script: tranche version ####
tranches_list <- list.files(dir_tranches)

for (fn_input in tranches_list){
	print(fn_input)
	
	Tranche <- read_parquet(file.path(dir_tranches, fn_input)) %>% lazy_dt
	
	fn_output <- str_replace(fn_input, 'tranche', 'comparable_metrics')
	dir_fn_output <- file.path(dir_output, fn_output)
	
	ComparableScalars <- get_comparable_scalars(
		Combos = Tranche, 
		FercSteam = FercSteam, 
		EiaPlantParts = EiaPlantParts
	)
	StopWordsPlantName <- get_stopwords_plantname(EiaPlantParts = EiaPlantParts)
	StopWordsUtilityName <- get_stopwords_utility_name(EiaPlantParts = EiaPlantParts)
	
	stop_words_plant_name_uber_pattern <- get_regex_uber_pattern(StopWords = StopWordsPlantName)
	stop_words_utility_name_uber_pattern <- get_regex_uber_pattern(StopWords = StopWordsUtilityName)
	
	StringMetrics <- get_string_metrics(
		Combos = Tranche,
		EiaPlantParts = EiaPlantParts,
		FercSteam = FercSteam,
		FercUtilities = FercUtilities,
		stop_words_plant_name_uber_pattern = stop_words_plant_name_uber_pattern,
		stop_words_utility_name_uber_pattern = stop_words_utility_name_uber_pattern
	)
	
	AdditionalColumns <- get_additional_columns(
		Combos = Tranche,
		FercSteam = FercSteam,
		EiaPlantParts = EiaPlantParts
	)
	
	Output <-
		Tranche %>%
		collect %>%
		bind_cols(StringMetrics) %>%
		bind_cols(ComparableScalars) %>%
		bind_cols(AdditionalColumns)
	
	Output %>% write_parquet(dir_fn_output)
}
