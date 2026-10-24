import pandas as pd

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')
    print(f"Data loaded. Shape: {df.shape}")
    ClinicalSignificanceFilter = ["Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic", "Pathogenic/Pathogenic"]
    df = df[df['ClinicalSignificance'].isin(ClinicalSignificanceFilter)]
    print(f"Filtered data. Shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    print("Run main.py instead")