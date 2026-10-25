import pandas as pd

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')

    print(f"Data loaded shape: {df.shape}")
    ClinicalSignificanceFilter = ["Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic", "Benign", "Benign/Likely benign", "Likely benign"]
    
    df = df[df['ClinicalSignificance'].isin(ClinicalSignificanceFilter)]
    print(f"Filtered data shape: {df.shape}")
    
    label_mapping = {
        "Pathogenic": 1,
        "Likely pathogenic": 1,
        "Pathogenic/Likely pathogenic": 1,
        "Benign": 0,
        "Benign/Likely benign": 0,
        "Likely benign": 0
    }
    df['target'] = df['ClinicalSignificance'].map(label_mapping)

    return df


if __name__ == "__main__":
    print("Run main.py instead")