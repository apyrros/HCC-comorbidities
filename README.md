# HCC condition classifier

Classifies 6 HCC, gender, age, and  RAF score based on frontal chest x-rays

Data directory structure set up as follows:
* raf_data
    * train.csv
    * test.csv
    * data

Usage:
* Training: python train.py --data_dir <path_to_data> --checkpoint_dir <checkpoint_save_directory> --size <size_of_xrays> -age_norm <normalization_for_age> --raf_norm <normalization_for_raf> --lr <learning_rate> --epochs <number_of_epochs> --train_batch_size <batch_size_for_training> --test_batch_size <batch_size_for_testing> --num_workers <number_of_workers> --decay_start_epoch <epoch_to_start_lr_decay>
* Testing: python test.py --data_dir <path_to_data> --checkpoint_path <path_to_checkpoint> --out_path <test_csv_location> --size <size_of_xrays> --age_norm <normalization_for_age> --raf_norm <normalization_for_raf> --only_pred <if_no_csv_file_is_used> --calc_stats <get_specificity_and_sensitivity>

Model weights available upon request to ayis@ayis.org for non-commercial use, research purposes only.
MIT license.
