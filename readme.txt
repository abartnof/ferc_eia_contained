
File Structure

```
scripts
├── general_scripts
	|── collect_data.R
	|── clean_positive_matches.R 
	|── create_training_matches_and_mismatches.R
├── model_a_gbm
	├── model_a_gbm_training
		├── model_a_gbm_training_create_comparable_metrics.R
	├── model_a_gbm_usage 
├── model_a_ann
	├── model_a_ann_training
	├── model_a_ann_usage 
├── model_b_gbm
	├── model_b_gbm_training
	├── model_b_gbm_usage 
├── model_b_ann
	├── model_b_ann_training
	├── model_b_ann_usage 

data
├── input_data
	|── eia_ferc1_training.csv *
	|── eia_plant_parts.RDS
	|── ferc_steam.RDS
	|── ferc_utilities.RDS
├── working_data
	|── positive_matches.RDS
	|── matches_and_mismatches.parquet
	├── model_a
		├── model_a_training
			├── all_joined_data.parquet
			├── stop_words_plant_name.csv
			├── stop_words_utility.csv
	├── model_b


```
Notes:
- The scripts are contained in the github repository, and can be placed anywhere you like. As they are simply scripts, they are rather small.
- The data, in contrast, can get quite large for this project. This is because a cartesian product has to be made, comparing each FERC entry with each possible EIA entry. (We can block out impossible matches by only comparing FERC and EIA entries from the same year, but the data still gets large.) Consequently, the data can be placed wherever it is most convenient for you: eg in your Documents folder, on an external hard drive, etc. At the top of each script, you'll note a variable called data_dir; please change this value to wherever you place the data.
- The Catalyst Co-op data comes from a very large sqlite file. This is pointed to explicitly in the collect_data.R file.
- The file input_data/eia_ferc1_training.csv was manually-created by RMI and Catalyst Co-Op, and represents human-created training matches. 


get all files:
find . -type f | pbcopy

# Initial ELT
	./general_scripts/collect_data.R
	./general_scripts/clean_positive_matches.R
	./general_scripts/create_training_matches_and_mismatches.R

# Model A training
	./model_a/model_a_training_create_comparable_metrics.R
	./model_a/model_a_training_feature_engineering.R

# Model A GBT Training
	./model_a/model_a_gbm/model_a_gbm_training/model_a_gbm_hyperparameter_search.ipynb
	./model_a/model_a_gbm/model_a_gbm_training/model_a_train_rank_hyperparameters.R
	./model_a/model_a_gbm/model_a_gbm_training/model_a_gbm_hyperparameters_search_2_cv.ipynb

# Model A ANN Training
	# NB Keras/tensorflow works best from BASH, so while scripts can be designed in jupyter notebooks, it's preferable to run the full scripts in BASH
	./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search.ipynb
	./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search.py
	./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search_dig_into_best_candidates.ipynb
	./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search_dig_into_best_candidates.py
	./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search_final_metrics_analysis.R


./.DS_Store
./general_scripts/collect_data.R
./general_scripts/create_training_matches_and_mismatches.R
./general_scripts/clean_positive_matches.R
./general_scripts/entire_dataset_cartesian_product_of_ferc_and_eia_ids.R
./model_a/.DS_Store
./model_a/model_a_ann/.DS_Store
./model_a/model_a_ann/model_a_ann_training/.DS_Store
./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search.ipynb
./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search.py
./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search_final_metrics_analysis.R
./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search_dig_into_best_candidates.ipynb
./model_a/model_a_ann/model_a_ann_training/model_a_ann_hyperparameter_search_dig_into_best_candidates.py
./model_a/model_a_ann/model_a_ann_training/.ipynb_checkpoints/model_a_ann_hyperparameter_search_dig_into_best_candidates-checkpoint.ipynb
./model_a/model_a_ann/model_a_ann_training/.ipynb_checkpoints/model_a_ann_hyperparameter_search-checkpoint.ipynb
./model_a/model_a_ann/model_a_ann_training/.ipynb_checkpoints/full_model_neural_network_hp_search-checkpoint.ipynb
./model_a/model_a_ann/model_a_ann_training/.ipynb_checkpoints/model_a_hyperparameter_search-checkpoint.ipynb
./model_a/model_a_ann/.ipynb_checkpoints/model_a_ann_hyperparameter_search_dig_into_best_candidates-checkpoint.ipynb
./model_a/model_a_ann/.ipynb_checkpoints/model_a_ann_hyperparameter_search-checkpoint.ipynb
./model_a/model_a_ann/.ipynb_checkpoints/full_model_neural_network_hp_search-checkpoint.ipynb
./model_a/model_a_ann/.ipynb_checkpoints/model_a_ann_hyperparameter_search-checkpoint.ipynb copy
./model_a/model_a_ann/.ipynb_checkpoints/copy model_a_ann_hyperparameter_search-checkpoint.ipynb
./model_a/model_a_ann/.ipynb_checkpoints/ray_tune-checkpoint.ipynb
./model_a/model_a_ann/.ipynb_checkpoints/model_a_hyperparameter_search-checkpoint.ipynb
./model_a/model_a_gbm/.DS_Store
./.ipynb_checkpoints/model_a-checkpoint.ipynb
./.ipynb_checkpoints/model_a_hyperparameter_search_dig_into_best_candidates-checkpoint.ipynb
./.ipynb_checkpoints/model_a_hyperparameter_search-checkpoint.ipynb
./dnu/.DS_Store
./dnu/model_a_ann_hyperparameter_search.ipynb
./dnu/full_model_neural_network_hp_search.ipynb
./dnu/model_a.ipynb
./dnu/model_a_ann_hyperparameter_search_final_metrics_analysis.R
./dnu/model_a_ann_hyperparameter_search_dig_into_best_candidates.ipynb
./dnu/model_a_hyperparameter_search.ipynb
./dnu/model_a_gbm_hyperparameter_search.py

