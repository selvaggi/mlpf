results_path="/eos/user/g/gkrzmanc/results/2024/test_1311_additional_features_300files"
export PYTHONPATH=$PYTHONPATH:${pwd}
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,take_out_pred_photons_0_1,take_out_fakes_only --output_dir take_out_pred_fake_photons_0_1 --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,take_out_pred_photons_1_10,take_out_fakes_only --output_dir take_out_pred_fake_photons_1_10 --mass-only
python src/evaluation/refactor/plot_results.py --path $results_path --preprocess class_correction,take_out_pred_photons_10_100,take_out_fakes_only --output_dir take_out_pred_fake_photons_10_100 --mass-only

