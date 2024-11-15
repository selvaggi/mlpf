
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path /eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_1 --preprocess class_correction,beta_correction --output_dir beta_correction
python src/evaluation/refactor/plot_results.py --path /eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_1 --preprocess class_correction --output_dir no_beta_correction

