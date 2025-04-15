library(tidyverse)
library(skimr)

#### ELT ####
data_dir <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/'

fn_feature_importance <- file.path(data_dir, 'working_data/model_second_stage/model_second_stage_training/temp/mod2_feature_importance.csv')
FeatureImportance <- read_csv(fn_feature_importance)
FeatureImportance <-
	FeatureImportance %>%
	mutate(
		group = case_when(
			str_detect(colnames, 'x1a') ~ 'Feature Encoding A',
			str_detect(colnames, 'x1b') ~ 'Feature Encoding B',
			colnames == 'y_fit__1_a_ann' ~ 'Y-Fit Feature Encoding A, ANN',
			colnames == 'y_fit__1_a_gbm' ~ 'Y-Fit Feature Encoding A, GBM',
			colnames == 'y_fit__1_b_ann' ~ 'Y-Fit Feature Encoding B, ANN',
			colnames == 'y_fit__1_b_gbm' ~ 'Y-Fit Feature Encoding B, GBM',
			colnames == 'y_fit_rank__1_a_ann' ~ 'Ranked Y-Fit Feature Encoding A, ANN',
			colnames == 'y_fit_rank__1_a_gbm' ~ 'Ranked Y-Fit Feature Encoding A, GBM',
			colnames == 'y_fit_rank__1_b_ann' ~ 'Ranked Y-Fit Feature Encoding B, ANN',
			colnames == 'y_fit_rank__1_b_gbm' ~ 'Ranked Y-Fit Feature Encoding B, GBM'
		),
		group = factor(group, ordered = FALSE)
	) %>%
	relocate(group, colnames, feature_importance)

MungedGroups <-
	FeatureImportance %>%
	group_by(group) %>%
	summarize(
		subtotal = sum(feature_importance)
	) %>%
	ungroup %>%
	mutate(
		prop = subtotal / sum(subtotal),
		label = scales::percent(prop, 1)
	) 

MungedGroups %>%
	mutate(group = fct_rev(group)) %>%
	ggplot(aes(x = group, y = prop, label = label)) +
	geom_col() +
	coord_flip() +
	scale_y_continuous(labels = scales::percent_format()) +
	labs(x = 'Feature', y = 'Importance', title = 'Second stage model feature importance') +
	theme(
		axis.ticks = element_blank()
	)
