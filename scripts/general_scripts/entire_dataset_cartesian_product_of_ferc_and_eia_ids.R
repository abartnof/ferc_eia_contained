# entire_dataset_cartesian_product_of_ferc_and_eia_ids.R
# Andrew Bartnof
# RMI, 2024

# Create a framework of each ferc record and eia record that will be compared. 
# This mirrors `create_matches_and_mismatches.R`. 
# Break this up into smaller files. Save on external hard drive. 
# Note that for now, this subsets to only 20 FERC records per year


library(tidyverse)
library(skimr)
library(dtplyr)
library(arrow)
set.seed(1)


data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_input <- file.path(data_dir, '/input_data')
dir_output <- file.path(data_dir, '/working_data/tranches_ferc_to_eia')

fn_eia_plant_parts <- file.path(dir_input, 'eia_plant_parts.RDS')
fn_ferc_steam <- file.path(dir_input, 'ferc_steam.RDS')

EiaPlantParts <- readRDS(fn_eia_plant_parts)
FercSteam <- readRDS(fn_ferc_steam)
max_group_size <- 20L

# Ensure that we only use report_years that are in both the FERC1 and EIA sets
SubtotalEia <-
	EiaPlantParts %>%
	count(report_year, name = 'n_eia')

SubtotalFerc <-
	FercSteam %>%
	count(report_year, name = 'n_ferc')

SubtotalsJoined <-
	SubtotalFerc %>%
	full_join(SubtotalEia, by = 'report_year') %>%
	mutate_at(c('n_ferc', 'n_eia'), replace_na, 0L) %>%
	mutate(year_min = pmin(n_ferc, n_eia)) %>%
	filter(year_min > 0)

relevant_report_years <- SubtotalsJoined$report_year

print('Relevant report years:'); print(relevant_report_years)

# the only two variables we need per table is record_id and year; make sure
# we have those, and omit everything else.
FercRecordsAndYears <-
	FercSteam %>%
	filter(report_year %in% relevant_report_years) %>%
	select(record_id_ferc1, report_year) %>%
	drop_na %>%
	arrange(report_year, record_id_ferc1)

EiaRecordsAndYears <-
	EiaPlantParts %>%
	filter(report_year %in% relevant_report_years) %>%
	select(record_id_eia, report_year) %>%
	drop_na %>%
	arrange(report_year, record_id_eia)

# Divide the FERC records into tranches, no larger than the 
# max_group_size, for each report_year
FercToTranche <-
	FercRecordsAndYears %>%
	group_by(report_year) %>%
	mutate(
		i = row_number() - 1L,  # Otherwise, the 1st tranche will be a bit smaller
		quotient = i / max_group_size,
		quotient_floored = floor(quotient),
		quotient_floored = str_pad(string = quotient_floored, width = 3, side = 'left', pad = '0')
	) %>%
	ungroup %>%
	mutate(tranche = str_c(report_year, quotient_floored, sep = '_')) %>%
	select(record_id_ferc1, report_year, tranche)

FercToTranche %>%
	head

# Iterate through tranches to create combos
tranche_list <- sort(unique(FercToTranche$tranche))

create_combos <- function(tranche_variable){
	print(tranche_variable)
	
	# fn_tranche <- str_c(dir_output, 'cartesian_product__', tranche_variable, '.parquet')
	incomplete_fn <- str_c('tranche__', tranche_variable, '.parquet')
	complete_fn <- file.path(dir_output, incomplete_fn)
	
	# Note the report year for this tranche
	report_year_variable <-
		FercToTranche %>%
		filter(tranche == tranche_variable) %>%
		slice(1) %>%
		pull(report_year)

	# Note the entire hypothesis space for EIA records to link to	
	RelevantEiaRecords <-
		EiaRecordsAndYears %>%
		filter(report_year == report_year_variable) %>%
		select(record_id_eia)

	# Cartesian product of the relevant FERC ids and all possible EIA ids	
	Output <-
		FercToTranche %>%
		filter(tranche == tranche_variable) %>%
		select(record_id_ferc1) %>%
		expand_grid(RelevantEiaRecords)
	
	Output %>%
		write_parquet(complete_fn)
}

for (t in tranche_list){
	create_combos(tranche_variable = t)
}
