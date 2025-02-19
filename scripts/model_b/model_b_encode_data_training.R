#### Source functions ####
dir_scripts <- '~/Documents/rmi/rematch_ferc_eia_pixi/ferc_eia/scripts/'
fn_functions <- file.path(dir_scripts, '/model_b/model_b_encode_data_functions.R')
source(fn_functions)

#### Load tables ####
Combo <- read_parquet(fn_combo)
FercSteam <- readRDS(fn_ferc_steam)
FercUtility <- readRDS(fn_ferc_utility)
EiaPlantParts <- readRDS(fn_eia_plant_parts)
FercToFold <- read_parquet(fn_ferc_to_fold)

ProductRecordIdFerc1ToIs100Percent <- read_parquet(fn_ferc1_is_100)
ProductRecordIdFerc1ToNum <- readRDS(fn_ferc1_to_numbers)
ProductPlantNameFerc1ToToken <- read_parquet(fn_ferc1_to_token)
ProductPlantNameLvDistance <- read_parquet(fn_plant_name_lv_distance)
ProductUtilityNameLvDistance <- read_parquet(fn_utility_name_lv_distance)


#### Call Functions ####

FercContext <- get_ferc_context(FercSteam, FercUtility)
EiaContext <- get_eia_context(EiaPlantParts)
PreppedCombo <- get_prepped_combo(
	Combo, 
	FercContext, 
	EiaContext, 
	ProductRecordIdFerc1ToIs100Percent, 
	ProductPlantNameFerc1ToToken,
	ProductPlantNameLvDistance,
	ProductUtilityNameLvDistance
)

recipe_fit_training <- get_recipe_fit_training(PreppedCombo)

X <- bake(recipe_fit_training, PreppedCombo)

Y <-
	Combo %>%
	select(is_match) %>%
	mutate(is_match = as.integer(is_match))

ID <-
	Combo %>%
	left_join(FercToFold, by = 'record_id_ferc1', relationship = 'many-to-many') %>%
	select(record_id_ferc1, record_id_eia, report_year, fold)

X %>% write_parquet(fn_training_x)
Y %>% write_parquet(fn_training_y)
ID %>% write_parquet(fn_training_id)