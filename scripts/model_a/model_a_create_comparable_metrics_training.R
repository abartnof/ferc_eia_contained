# Run the functions to create the comparable metrics, as sourced
# from another file
# version: training data

# Use these locations to make the script work on your machine
dir_scripts <- '~/Documents/rmi/rematch_ferc_eia_pixi/ferc_eia/scripts/'
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

# Source script
fn_script <- file.path(dir_scripts, '/model_a/model_a_create_comparable_metrics_functions.R')
source(fn_script)

# Point to other files
dir_input <- file.path(data_dir, '/input_data/')
dir_working <- file.path(data_dir, '/working_data/')
dir_working_model_a_training <- file.path(
	data_dir,
	'/working_data/model_a/model_a_training/'
)

fn_eia_plant_parts <- file.path(dir_input, 'eia_plant_parts.RDS')
fn_ferc_steam <- file.path(dir_input, 'ferc_steam.RDS')
fn_ferc_utilities <- file.path(dir_input, 'ferc_utilities.RDS')
fn_matches_and_mismatches <- file.path(dir_working, 'matches_and_mismatches.parquet')

fn_all_joined_data <- file.path(dir_working_model_a_training, 'all_joined_data.parquet')
fn_stop_words_plant_name <- file.path(dir_working_model_a_training, 'stop_words_plant_name.csv')
fn_stop_words_utility_name <- file.path(dir_working_model_a_training, 'stop_words_utility.csv')

# Load files
FercSteam <- readRDS(fn_ferc_steam) %>% lazy_dt
FercUtilities <- readRDS(fn_ferc_utilities) %>% lazy_dt
EiaPlantParts <- readRDS(fn_eia_plant_parts) %>% lazy_dt
Combos <- read_parquet(fn_matches_and_mismatches) %>% lazy_dt

#### Run Script: training version ####
ComparableScalars <- get_comparable_scalars(
	Combos = Combos, 
	FercSteam = FercSteam, 
	EiaPlantParts = EiaPlantParts
)
StopWordsPlantName <- get_stopwords_plantname(EiaPlantParts = EiaPlantParts)
StopWordsUtilityName <- get_stopwords_utility_name(EiaPlantParts = EiaPlantParts)

# Run if training, as note-keeping:
StopWordsPlantName %>%
	write_csv(fn_stop_words_plant_name)
StopWordsUtilityName %>%
	write_csv(fn_stop_words_utility_name)

stop_words_plant_name_uber_pattern <- get_regex_uber_pattern(StopWords = StopWordsPlantName)
stop_words_utility_name_uber_pattern <- get_regex_uber_pattern(StopWords = StopWordsUtilityName)

StringMetrics <- get_string_metrics(
	Combos = Combos,
	EiaPlantParts = EiaPlantParts,
	FercSteam = FercSteam,
	FercUtilities = FercUtilities,
	stop_words_plant_name_uber_pattern = stop_words_plant_name_uber_pattern,
	stop_words_utility_name_uber_pattern = stop_words_utility_name_uber_pattern
)

AdditionalColumns <- get_additional_columns(
	Combos = Combos,
	FercSteam = FercSteam,
	EiaPlantParts = EiaPlantParts
)

Output <-
	Combos %>%
	collect %>%
	bind_cols(StringMetrics) %>%
	bind_cols(ComparableScalars) %>%
	bind_cols(AdditionalColumns)

Output %>% write_parquet(fn_all_joined_data)