results_path="/eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_300files"
#results_path="/eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_1"
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction --output_dir no_beta_correction --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,beta_correction --output_dir beta_correction --mass-only

#python src/evaluation/refactor/plot_results.py --path   /eos/user/g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss_300files --preprocess class_correction --output_dir reprod_plots_15_11_

results_path="/eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_1"
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction --output_dir no_beta_correction --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,beta_correction --output_dir beta_correction --mass-only

results_path="/eos/user/g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss_300files"
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction --output_dir no_beta_correction --mass-only


#python src/evaluation/refactor/plot_results.py --path   /eos/user/g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss_300files --preprocess class_correction --output_dir reprod_plots_15_11_