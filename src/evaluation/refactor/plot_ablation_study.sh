results_path="/eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_300files"
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_E --output_dir ablation_study_E --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_p --output_dir ablation_study_p --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_pid --output_dir ablation_study_pid --mass-only

results_path="/eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_1"
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_E --output_dir ablation_study_E --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_p --output_dir ablation_study_p --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_pid --output_dir ablation_study_pid --mass-only

results_path="/eos/user/g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss_300files"
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_E --output_dir ablation_study_E --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_p --output_dir ablation_study_p --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,ablation_study_pid --output_dir ablation_study_pid --mass-only

