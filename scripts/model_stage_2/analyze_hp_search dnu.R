library(tidyverse)
library(skimr)
library(car)

fn1 <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_02_21/gbm_grid_2025_02_21.csv'
fn2 <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_03_01/gbm_grid_2025_03_01.csv'
fn3 <- '/Volumes/Extreme SSD/rematch_eia_ferc1_docker/working_data/model_second_stage/model_second_stage_training/gbm_raytune_2025_03_02/gbm_grid_2025_03_02.csv'

clean_data <- function(X){
	X %>%
	select(
		rank, 
		order, 
		num_trees = `config/num_trees`,
		learning_rate = `config/learning_rate`,
		min_data_in_leaf = `config/min_data_in_leaf`,
		binary_logloss, 
		auc
	) %>%
	mutate_at(c('rank', 'order', 'num_trees', 'min_data_in_leaf'), as.integer)
}

HP1 <- read_csv(fn1) %>% clean_data %>% mutate(run = 1L)
HP2 <- read_csv(fn2) %>% clean_data %>% mutate(run = 2L)
HP3 <- read_csv(fn3) %>% clean_data %>% mutate(run = 3L)

JoinedData <-
	HP1 %>%
	bind_rows(HP2) %>%
	bind_rows(HP3) %>%
	relocate(run) %>%
	mutate(run = ordered(run))

scatterplotMatrix(~ num_trees + learning_rate + min_data_in_leaf + binary_logloss | run, data = JoinedData, plot.points = FALSE)

JoinedData %>%
	select(run, num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%
	gather(variable, value, -run) %>%
	ggplot(aes(x = run, y = value)) +
	geom_boxplot() + 
	facet_wrap(~variable, scales = 'free')

JoinedData %>%
	select(run, num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%
	gather(variable, value, -run) %>%
	group_by(run, variable) %>%
	summarize(
		q50 = median(value),
		q25 = quantile(value, 0.25),
		q75 = quantile(value, 0.75),
		q025 = quantile(value, 0.025),
		q975 = quantile(value, 0.975),
	) %>%
	ungroup %>%
	ggplot(aes(x = run)) +
	geom_linerange(aes(ymin = q025, ymax = q975), linewidth = 5, color = 'black') +
	geom_linerange(aes(ymin = q25, ymax = q75), linewidth = 5, color = '#00843D') +
	geom_point(aes(y = q50), color = '#FFCD00') +
	facet_wrap(~variable, scales = 'free') +
	labs(y = '', x = 'Bootstrap #')

# Method: BEST
# Pull the 10 best models from each run, remove duplicates
JoinedData
	select(run, rank, num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%


# Method: AVG


HP1 %>%
	bind_rows(HP2) %>%
	select(run, num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%
	gather(variable, value, -run) %>%
	ggplot(aes(x = value, group = run, color = run)) +
	geom_density() +
	facet_wrap(~variable, scales = 'free')

HP1 %>%
	bind_rows(HP2) %>%
	select(run, num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%
	gather(variable, value, -run) %>%
	mutate(run = factor(run)) %>%
	ggplot(aes(x = run, y = value, color = run)) +
	geom_boxplot(outlier.alpha = 0.5) +
	facet_wrap(~variable, scales = 'free')


HP1 %>%
	bind_rows(HP2) %>%
	with(., t.test(num_trees ~ run))

HP1 %>%
	bind_rows(HP2) %>%
	with(., wilcox.test(formula = num_trees ~ run))
	
HP1 %>%
	bind_rows(HP2) %>%
	with(., wilcox.test(formula = learning_rate ~ run))

HP1 %>%
	bind_rows(HP2) %>%
	with(., wilcox.test(formula = min_data_in_leaf ~ run))

HP1	%>%
	select(num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%
	scatterplotMatrix()

HP1	%>%
	select(num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%
	scatterplotMatrix()

JoinedData <-
	HP1	%>%
	bind_rows(HP2) %>%
	select(run, num_trees, learning_rate, min_data_in_leaf, binary_logloss) %>%
	mutate(run = factor(run))

scatterplotMatrix(~ income + education + prestige | type, data=Duncan)
scatterplotMatrix(~ num_trees + learning_rate + min_data_in_leaf + binary_logloss | run, data = JoinedData, plot.points = FALSE)
