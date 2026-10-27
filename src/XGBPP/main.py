import sys
import data_loader as data_loader
import data_processing as data_processing
import evaluate as evaluate
import explain as explain

def main():
    df = data_loader.load_data('./data/variant_summary.txt')
    # Running EDA
    data_processing.run_eda(df)

    # Building base features
    Xdf = data_processing.build_base_features(df)

    # Splitting the dataset into training and testing sets
    train_df, test_df = data_processing.perform_gene_grouped_split(df, 0.2, 42)

    # Extracting features and labels for training
    X_train = data_processing.build_base_features(train_df)
    y_train = train_df['target'].values
    X_test = data_processing.build_base_features(test_df)

    # X_train = X_train.drop(columns=["position"])
    # X_test = X_test.drop(columns=["position"])

    evaluate.evaluate_baseline(X_train, y_train)

    explain.generate_shap_explanations(X_train, y_train, X_test)


if __name__ == "__main__":
    main()