# Iterate through the y-fit values from models a and b, to see how much 
# they (dis)agree with each other
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof

library(tidyverse)
library(skimr)
library(arrow)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'
dir_tranches <- file.path(data_dir, 'working_data/tranches_ferc_to_eia')
dir_model_a_ann_y_fit <- file.path(data_dir, '/working_data/model_a/model_a_ann_y_fit')
dir_model_a_gbm_y_fit <- file.path(data_dir, '/working_data/model_a/model_a_gbm_y_fit')
dir_model_b_ann_y_fit <- file.path(data_dir, '/working_data/model_b/model_b_ann_y_fit')
dir_model_b_gbm_y_fit <- file.path(data_dir, '/working_data/model_b/model_b_gbm_y_fit')

fn_cor <- file.path(data_dir, '/working_data/model_second_stage/model_a_b_cor.parquet')
fn_misc_stats <- file.path(data_dir, '/working_data/model_second_stage/model_a_b_misc_stats.parquet')


# Locate tranche ID and y_fit files 

CteTrancheFn <-
	list.files(dir_tranches, full.names = T) %>%
	enframe(name = NULL, value = 'fn_tranche') %>%
	mutate(
		id = str_replace(fn_tranche, '^.*tranche__', ''),
		id = str_replace(id, '\\.parquet', '')
	)

CteModelAAnnYFit <-
	list.files(dir_model_a_ann_y_fit, full.names = T) %>%
	enframe(name = NULL, value = 'fn_model_a_ann_y_fit') %>%
	mutate(
		id = str_replace(fn_model_a_ann_y_fit, '^.*y_fit__', ''),
		id = str_replace(id, '\\.parquet', ''),
		# model = 'ANN A'
	)

CteModelAGbmYFit <-
	list.files(dir_model_a_gbm_y_fit, full.names = T) %>%
	enframe(name = NULL, value = 'fn_model_a_gbm_y_fit') %>%
	mutate(
		id = str_replace(fn_model_a_gbm_y_fit, '^.*y_fit__', ''),
		id = str_replace(id, '\\.parquet', ''),
		# model = 'GBM A'
	)

CteModelBAnnYFit <-
	list.files(dir_model_b_ann_y_fit, full.names = T) %>%
	enframe(name = NULL, value = 'fn_model_b_ann_y_fit') %>%
	mutate(
		id = str_replace(fn_model_b_ann_y_fit, '^.*y_fit__', ''),
		id = str_replace(id, '\\.parquet', ''),
		# model = 'ANN B'
	)

CteModelBGbmYFit <-
	list.files(dir_model_b_gbm_y_fit, full.names = T) %>%
	enframe(name = NULL, value = 'fn_model_b_gbm_y_fit') %>%
	mutate(
		id = str_replace(fn_model_b_gbm_y_fit, '^.*y_fit__', ''),
		id = str_replace(id, '\\.parquet', ''),
		# model = 'GBM B'
	)

FN <-
	CteTrancheFn %>%
	full_join(CteModelAAnnYFit, by = join_by(id)) %>%
	full_join(CteModelAGbmYFit, by = join_by(id)) %>%
	full_join(CteModelBAnnYFit, by = join_by(id)) %>%
	full_join(CteModelBGbmYFit, by = join_by(id))

# Iterate
CollectedCor <- tibble()
CollectedMiscStats <- tibble()
for (i in seq(1, nrow(FN))){
	print(sprintf('%s out of %s', i, nrow(FN)))
	
	Tranche <- read_parquet(FN$fn_tranche[i])
	ModelAAnnYFit <- read_parquet(FN$fn_model_a_ann_y_fit[i]) %>%
		rename(y_fit_a_ann = y_fit)
	ModelAGbmYFit <- read_parquet(FN$fn_model_a_gbm_y_fit[i]) %>%
		rename(y_fit_a_gbm = y_fit)
	ModelBAnnYFit <- read_parquet(FN$fn_model_b_ann_y_fit[i]) %>%
		rename(y_fit_b_ann = y_fit)
	ModelBGbmYFit <- read_parquet(FN$fn_model_b_gbm_y_fit[i]) %>%
		rename(y_fit_b_gbm = y_fit)

	JoinedData <-
		Tranche %>%
		bind_cols(ModelAAnnYFit) %>%
		bind_cols(ModelAGbmYFit) %>%
		bind_cols(ModelBAnnYFit) %>%
		bind_cols(ModelBGbmYFit)

	# Look at the correlations between the models	
	Cor <-
		JoinedData %>%
		select(record_id_ferc1, starts_with('y_fit')) %>%
		group_by(record_id_ferc1) %>%
		nest %>%
		mutate(
			data = map(data, as.matrix),
			cor = map(data, cor, method='p'),
			cor = map(cor, as.data.frame),
			cor = map(cor, rownames_to_column)
		) %>%
		unnest(cor) %>%
		ungroup %>%
		select(record_id_ferc1, rowname, starts_with('y_fit'))
	CollectedCor <- CollectedCor %>% bind_rows(Cor)

	# Collect interesting stats about how the y_fits look
	MiscStats <-
		JoinedData %>%
		select(record_id_ferc1, starts_with('y_fit')) %>%
		gather(model, y_fit, -record_id_ferc1) %>%
		group_by(record_id_ferc1, model) %>%
		summarize(
			mean_y_fit = mean(y_fit),
			min_y_fit = min(y_fit),
			q025 = quantile(y_fit, 0.025),
			q25 = quantile(y_fit, 0.25),
			median_y_fit = median(y_fit),
			q75 = quantile(y_fit, 0.75),
			q975 = quantile(y_fit, 0.975),
			max_y_fit = max(y_fit),
			num_ones = sum(near(y_fit, 1.0)),
			num_zeroes = sum(near(y_fit, 0.0)),
			n = n()
		) %>%
		ungroup
	CollectedMiscStats <- CollectedMiscStats %>% bind_rows(MiscStats)
}

CollectedCor %>% write_parquet(fn_cor)
CollectedMiscStats %>% write_parquet(fn_misc_stats)