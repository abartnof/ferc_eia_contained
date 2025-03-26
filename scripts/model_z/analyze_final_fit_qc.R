library(tidyverse)
library(arrow)
library(skimr)

#### ELT ####
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
fn_mappings <- file.path(data_dir, '/output_data/top_mappings.parquet')
Mappings <- read_parquet(fn_mappings) %>% 
	relocate(record_id_ferc1)

# fn_matches <- file.path(data_dir, 'working_data/positive_matches.RDS')
fn_ferc_steam <- file.path(data_dir, 'input_data/ferc_steam.RDS')
fn_ferc_utilities <- file.path(data_dir, 'input_data/ferc_utilities.RDS')
fn_eia <- file.path(data_dir, 'input_data/eia_plant_parts.RDS')



FercSteam <- readRDS(fn_ferc_steam)
FercUtilities <- readRDS(fn_ferc_utilities)
Eia <- readRDS(fn_eia)

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

# 
Mappings %>%
	left_join(CteFerc, by = 'record_id_ferc1') %>%
	left_join(CteEia, by = 'record_id_eia') %>%
	mutate(
		ratio_ferc_to_eia_net_gen = round(net_generation_mwh_ferc1/net_generation_mwh_eia, 2)
	) %>%
	write_csv('~/Downloads/mappings_qc.csv')
