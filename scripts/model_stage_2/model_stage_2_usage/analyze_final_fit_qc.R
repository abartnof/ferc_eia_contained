library(tidyverse)
library(arrow)
library(skimr)

#### ELT ####
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
fn_mappings <- file.path(data_dir, '/output_data/top_mappings.parquet')
Mappings <- read_parquet(fn_mappings) %>% 
	relocate(record_id_ferc1) %>%
	mutate(y_fit = round(y_fit, 3))
Mappings

fn_cc <- '/Users/andrewbartnof/Downloads/out_pudl__yearly_assn_eia_ferc1_plant_parts.parquet'
CC <- read_parquet(fn_cc)
CteCC <-
	CC %>%
	select(record_id_ferc1, record_id_eia, match_type)

fn_ferc_steam <- file.path(data_dir, 'input_data/ferc_steam.RDS')
fn_ferc_utilities <- file.path(data_dir, 'input_data/ferc_utilities.RDS')
fn_eia <- file.path(data_dir, 'input_data/eia_plant_parts.RDS')

FercSteam <- readRDS(fn_ferc_steam)
FercUtilities <- readRDS(fn_ferc_utilities)
Eia <- readRDS(fn_eia)

# Moderate data cleaning, joining

CteEia <-
	Eia %>%
	select(record_id_eia, plant_part, plant_name_eia, utility_name_eia, net_generation_mwh, capacity_mw, technology_description) %>%
	rename(plant_part_eia = plant_part, net_generation_mwh_eia = net_generation_mwh, capacity_mw_eia = capacity_mw, technology_description_eia = technology_description) %>%
	relocate(record_id_eia, plant_name_eia, plant_part_eia, utility_name_eia)

CteFerc <-
	FercSteam %>%
	select(record_id_ferc1, plant_name_ferc1, utility_id_ferc1, net_generation_mwh, capacity_mw, plant_type) %>%
	left_join(FercUtilities, by = 'utility_id_ferc1') %>%
	select(record_id_ferc1, plant_name_ferc1, utility_name_ferc1, net_generation_mwh, capacity_mw, plant_type) %>%
	rename(net_generation_mwh_ferc1 = net_generation_mwh, capacity_mw_ferc1 = capacity_mw, plant_type_eia = plant_type)
# 
# FinalCC <-
	CteCC %>%
	left_join(CteFerc, by = 'record_id_ferc1') %>%
	left_join(CteEia, by = 'record_id_eia') %>%
	mutate(
		ratio_ferc_to_eia_net_gen = round(net_generation_mwh_ferc1/net_generation_mwh_eia, 2),
		ratio_ferc_to_eia_capacity = round(capacity_mw_ferc1/capacity_mw_eia, 2)
	) %>%
	mutate_if(is.character, replace_na, '(Missing)') %>%
	rename_with(.fn = function(.x){paste0(.x,"_pudl")}) %>%
	rename(record_id_ferc1 = record_id_ferc1_pudl)
#
FinalRmi <-
	Mappings %>%
	left_join(CteFerc, by = 'record_id_ferc1') %>%
	left_join(CteEia, by = 'record_id_eia') %>%
	mutate(
		ratio_ferc_to_eia_net_gen = round(net_generation_mwh_ferc1/net_generation_mwh_eia, 2),
		ratio_ferc_to_eia_capacity = round(capacity_mw_ferc1/capacity_mw_eia, 2)
	) %>%
	rename_with(.fn = function(.x){paste0(.x,"_rmi")}) %>%
	rename(record_id_ferc1 = record_id_ferc1_rmi)

FinalRmi %>%
	left_join(FinalCC, by = 'record_id_ferc1') %>%
	mutate_if(is.numeric, round, 3) %>%
	write_csv('~/Downloads/mappings_qc.csv')
# 
# 
# 
# # 
# Cte1 <-
# 	CC %>%
# 	select(record_id_ferc1, record_id_eia, match_type) %>%
# 	mutate(
# 		match_type = fct_explicit_na(match_type),
# 		record_id_eia = fct_explicit_na(record_id_eia)
# 	) %>%
# 	rename(record_id_eia_pudl = record_id_eia)
# 
# Cte2 <-
# 	Mappings %>%
# 	select(record_id_ferc1, record_id_eia) %>%
# 	rename(record_id_eia_rmi = record_id_eia)
# 
# Cte2 %>%
# 	left_join(Cte1, by = 'record_id_ferc1') %>%
# 	mutate(is_match = record_id_eia_rmi == record_id_eia_pudl) %>%
# 	group_by(match_type) %>%
# 	summarize(prop = mean(is_match)) %>%
# 	ungroup
# 
# CC %>%
# 	mutate(
# 		match_type = fct_explicit_na(match_type)
# 	) %>%
# 	count(match_type) %>%
# 	mutate(prop = prop.table(n))
