# Analyze the previously collected analytical data comparing the y-fits 
# from models a and b
# author: Andrew Bartnof
# copyright: Copyright 2025, Rocky Mountain Institute
# credits: Alex Engel, Andrew Bartnof

library(tidyverse)
library(skimr)
library(arrow)
library(corrplot)

data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_cor <- file.path(data_dir, '/working_data/model_second_stage/model_a_b_cor.parquet')
fn_misc_stats <- file.path(data_dir, '/working_data/model_second_stage/model_a_b_misc_stats.parquet')
Cor <- read_parquet(fn_cor)
MiscStats <- read_parquet(fn_misc_stats)

fn_cor_image <- file.path(data_dir, '/output_data/stats/median_pearsons_cor.png')

# Correlations as boxplots

# Cor %>%
# 	rename(model1 = rowname) %>%
# 	gather(model2, rho, -record_id_ferc1, -model1) %>%
# 	drop_na %>%
# 	mutate_at(c('model1', 'model2'), str_replace, 'y_fit_', '') %>%
# 	mutate_at(c('model1', 'model2'), str_replace_all, '_', ' ') %>%
# 	mutate_at(c('model1', 'model2'), str_to_title) %>%
# 	mutate_at(c('model1', 'model2'), str_replace_all, 'Ann', 'ANN') %>%
# 	mutate_at(c('model1', 'model2'), str_replace_all, 'Gbm', 'GBM') %>%
# 	ggplot(aes(x = 1, y = rho)) +
# 	geom_boxplot(width = 0.75) +
# 	geom_hline(yintercept = 0, linetype = 'dashed', color = 'darkgrey') +
# 	facet_grid(model1 ~ model2, switch = 'y') +
# 	scale_y_continuous(limits = c(-1, 1)) +
# 	scale_x_continuous(limits = c(0, 2)) +
# 	theme_light() +
# 	labs(x = '', y = 'Spearman\'s Rho')

Matrix <-
	Cor %>%
	rename(model1 = rowname) %>%
	gather(model2, rho, -record_id_ferc1, -model1) %>%
	mutate_at(c('model1', 'model2'), str_replace, 'y_fit_', '') %>%
	mutate_at(c('model1', 'model2'), str_replace_all, '_', ' ') %>%
	mutate_at(c('model1', 'model2'), str_to_title) %>%
	drop_na(rho) %>%
	group_by(model1, model2) %>%
	summarize(median_rho = median(rho)) %>%
	ungroup %>%
	spread(model2, median_rho)
Matrix <- as.data.frame(Matrix)
rownames(Matrix) <- Matrix$model1
Matrix$model1 <- NULL
Matrix <- as.matrix(Matrix)

png(filename=fn_cor_image); corrplot(Matrix, diag=TRUE, method='color', type='lower', addCoef.col='black'); dev.off()
corrplot(Matrix, diag=TRUE, method='color', type='lower', addCoef.col='black')

#
# MiscStats %>%
# 	select(record_id_ferc1, model, num_ones) %>%
# 	mutate(does_contain_a_one = num_ones > 0L) %>%
# 	group_by(model) %>%
# 	summarize(prop_contains_a_one = mean(does_contain_a_one)) %>%
# 	ungroup
# 
# # model_a_ann has a lot of ones
# MiscStats %>%
# 	select(model, num_ones) %>%
# 	ggplot(aes(x = num_ones)) +
# 	geom_histogram() +
# 	facet_wrap(~model)
