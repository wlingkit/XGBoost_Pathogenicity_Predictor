# XGBoost_Pathogenicity_Predictor

# Readings
https://www.mcb111.org/w06/durbin_book.pdf
https://www.nvidia.com/en-us/glossary/xgboost/

# Downloading test data
https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

# Baseline Results
--- Logistic Regression Baseline ---
Mean ROC-AUC: 0.5212 | Mean PR-AUC: 0.2590

--- XGBoost Model ---
Mean ROC-AUC: 0.6356 | Mean PR-AUC: 0.3461

# SHAP summary
"Position" had the biggest impact on the prediction while len_diff, alt_len, ref_len, and is_transition were contributing almost nothing. This doesn't mean the project was done incorrectly. It does suggests that the dataset and the feauters were inefficient to make an accurate prediction. This suggests we need better features for predicting pathogenicity.