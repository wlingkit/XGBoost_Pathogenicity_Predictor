import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def evaluate_baseline(X: pd.DataFrame, y: np.ndarray) -> None:
    # Set up Stratified K-Fold cross-validation
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_aucs, xgb_aucs = [], []
    lr_aps, xgb_aps = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        lr_pipeline.fit(X_train, y_train)
        lr_preds = lr_pipeline.predict_proba(X_val)[:, 1]
        
        lr_aucs.append(roc_auc_score(y_val, lr_preds))
        lr_aps.append(average_precision_score(y_val, lr_preds))

        # --- Baseline 2: XGBoost ---
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_weight,
            eval_metric='logloss',
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict_proba(X_val)[:, 1]

        xgb_aucs.append(roc_auc_score(y_val, xgb_preds))
        xgb_aps.append(average_precision_score(y_val, xgb_preds))
    
    print(f"--- Logistic Regression Baseline ---")
    print(f"Mean ROC-AUC: {np.mean(lr_aucs):.4f} | Mean PR-AUC: {np.mean(lr_aps):.4f}")
    
    print(f"\n--- XGBoost Model ---")
    print(f"Mean ROC-AUC: {np.mean(xgb_aucs):.4f} | Mean PR-AUC: {np.mean(xgb_aps):.4f}")

if __name__ == "__main__":
    pass