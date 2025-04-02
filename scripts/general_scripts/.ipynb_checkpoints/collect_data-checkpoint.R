#-------------------------------------------------------------------------------
# collect_data.R

# Extract tables from Catalyst Coöp PUDL sqlite files, 
# perform light data cleaning.

# Author: Andrew Bartnof, for RMI
# Email: abartnof.contractor@rmi.org
# 2024

# input: 
# 	CC PUDL Sqlite file
# output: 
# 	EIA plant parts table
# 	FERC1 table
# 	FERC utility table (map utility_id_ferc to utility_name_ferc)
#-------------------------------------------------------------------------------

library(tidyverse)
library(skimr)
library(dplyr)
library(RSQLite)
library(lubridate)

fn_pudl_sqlite <- '/Volumes/Extreme SSD/pudl.sqlite'

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker'
fn_eia_plant_parts <- file.path(data_dir, '/input_data/', 'eia_plant_parts.RDS')
fn_ferc_steam <- file.path(data_dir, '/input_data/', 'ferc_steam.RDS')
fn_ferc_utilities <- file.path(data_dir, '/input_data/', 'ferc_utilities.RDS')

con <- DBI::dbConnect(RSQLite::SQLite(), dbname = fn_pudl_sqlite)

#### Eia yearly plant parts ####
EiaQuery <- tbl(con, "out_eia__yearly_plant_parts")

EiaYearlyPlantParts <-
	EiaQuery %>%
	filter(
		true_gran == 1L,
		ownership_dupe == 0L, # New, as of c. 2024_05_16
		# technology_description != 'Conventional Hydroelectric',
		# technology_description != 'Solar Photovoltaic',
		# technology_description != 'Onshore Wind Turbine',
		# technology_description != 'Hydroelectric Pumped Storage',
		# technology_description != 'Batteries',
		# technology_description != 'Offshore Wind Turbine',
		# technology_description != 'Flywheels'
	) %>%
	as_tibble

EiaYearlyPlantPartsClean <-
	EiaYearlyPlantParts %>%
	mutate_at(vars(contains('_id')), as.character) %>%
	mutate_at(vars(contains('ownership_dupe')), as.logical) %>%
	mutate_at('report_date', lubridate::as_date) %>% 
	mutate(
		plant_name_eia = str_to_lower(plant_name_eia),
		utility_name_eia = str_to_lower(utility_name_eia),
	) %>%
	rowid_to_column()

EiaYearlyPlantPartsClean %>%
	saveRDS(fn_eia_plant_parts)

#### FERC steam table ####
FercSteam <- 
	tbl(con, "core_ferc1__yearly_steam_plants_sched402") %>%
	collect()

FercSteamClean <-
	FercSteam %>%
	mutate_at(vars(contains('_id')), as.character) %>%
	mutate_at('plant_name_ferc1', str_to_lower) %>%
	rename(record_id_ferc1 = record_id) %>%
	rowid_to_column

FercSteamClean %>%
	saveRDS(fn_ferc_steam)

#### Map FERC utility name to utility id ####
FercUtilities <-
	tbl(con, 'out_ferc1__yearly_all_plants') %>%
	distinct(utility_id_ferc1, utility_name_ferc1) %>%
	mutate(utility_id_ferc1 = as.character(utility_id_ferc1)) %>%
	mutate(utility_name_ferc1 = str_to_lower(utility_name_ferc1)) %>%
	collect

FercUtilities %>%
	saveRDS(fn_ferc_utilities)

dbDisconnect(con)
