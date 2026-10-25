import sys
import data_loader as data_loader
import data_processing as data_processing

def main():
    df = data_loader.load_data('./data/variant_summary_50.txt')
    # Running EDA
    data_processing.run_eda(df)

    # Building base features
    Xdf = data_processing.build_base_features(df)

    # Splitting the dataset into training and testing sets
    train_df, test_df = data_processing.perform_gene_grouped_split(df, 0.2, 42)

if __name__ == "__main__":
    main()