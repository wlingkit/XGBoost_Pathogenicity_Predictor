import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_shap_explanations(X_train, y_train, X_test):
    print("\nTraining XGBoost model for explanation...")
    scale_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)

    print("\nInitializing SHAP TreeExplainer and calculating values...")
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    print("SHAP values calculated successfully.")

    print("\nGenerating SHAP Summary Plot...")
    
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    
    plt.title("XGBoost Baseline Feature Importance (SHAP)", fontsize=16)
    plt.tight_layout()
    
    output_image = "results/shap_summary_baseline.png"
    plt.savefig(output_image, dpi=300)
    print(f"Summary plot saved to: {output_image}")
    
    plt.show()

if __name__ == "__main__":
    pass