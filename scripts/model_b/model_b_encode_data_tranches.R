#### Source functions ####
dir_scripts <- '~/Documents/rmi/rematch_ferc_eia_pixi/ferc_eia/scripts/model_b'
fn_functions <- file.path(dir_scripts, 'model_b_encode_data_functions.R')
source(fn_functions)

dir_tranches <- file.path(data_dir, 'working_data/tranches_ferc_to_eia')
dir_x <- file.path(dir_working, '/model_b_x/')


#### Load tables ####
Combo <- read_parquet(fn_combo)
FercSteam <- readRDS(fn_ferc_steam)
FercUtility <- readRDS(fn_ferc_utility)
EiaPlantParts <- readRDS(fn_eia_plant_parts)

ProductRecordIdFerc1ToIs100Percent <- read_parquet(fn_ferc1_is_100)
ProductRecordIdFerc1ToNum <- readRDS(fn_ferc1_to_numbers)
ProductPlantNameFerc1ToToken <- read_parquet(fn_ferc1_to_token)
ProductPlantNameLvDistance <- read_parquet(fn_plant_name_lv_distance)
ProductUtilityNameLvDistance <- read_parquet(fn_utility_name_lv_distance)

#### Call Functions ####

# Train the recipe on the entire training data, including learning
# to scale the data/replace missing values with zeroes
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
recipe_fit_tranches <- get_recipe_fit_tranches(PreppedCombo)

# Use this feature engineering process to iterate through the tranches
FN <-
	list.files(dir_tranches, full.names = FALSE) %>%
	enframe(name = 'i', value = 'fn_tranche') %>%
	mutate(
		fn_x = str_replace(fn_tranche, 'tranche', 'x'),
		dir_fn_x = file.path(dir_x, fn_x),
		dir_fn_tranche = file.path(dir_tranches, fn_tranche)
	)

for (i in FN$i){
	print(
		sprintf("%i of %i", i, max(FN$i))
	)
	Tranche <- read_parquet(FN$dir_fn_tranche[i])
	PreppedTranche <- get_prepped_combo(
		Tranche, 
		FercContext, 
		EiaContext, 
		ProductRecordIdFerc1ToIs100Percent, 
		ProductPlantNameFerc1ToToken,
		ProductPlantNameLvDistance,
		ProductUtilityNameLvDistance
	)
	XTranche <- bake(recipe_fit_tranches, PreppedTranche)
	XTranche %>% write_parquet(FN$dir_fn_x[i])
}
